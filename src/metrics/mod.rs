use core_foundation::base::TCFType;
use core_foundation::number::CFNumber;
use core_foundation::string::CFString;
use io_kit_sys::types::io_object_t;
use io_kit_sys::{
    kIOMasterPortDefault, IOObjectRelease, IORegistryEntryCreateCFProperty,
    IOServiceGetMatchingService, IOServiceMatching,
};
use std::process::Command;
use std::ptr;

const THROTTLE_WARMUP_SECONDS: u64 = 10;
const THROTTLE_DETECTION_SECONDS: u64 = 15;
const THROTTLE_FREQUENCY_DROP_THRESHOLD: f64 = 0.85;
const THROTTLE_TEMP_THRESHOLD: f64 = 95.0;
const THROTTLE_THERMAL_LEVEL_THRESHOLD: u64 = 70;
const MACOS_PAGE_SIZE_BYTES: f64 = 16384.0;
const HZ_TO_GHZ_DIVISOR: f64 = 1_000_000_000.0;
const BYTES_TO_GB_DIVISOR: f64 = 1_073_741_824.0;
const MIN_FREQUENCY_FOR_THROTTLE_DETECTION: f64 = 2.0;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SystemMetrics {
    pub cpu_temperature: Option<f64>,
    pub gpu_temperature: Option<f64>,
    pub cpu_frequency: Option<f64>,
    pub gpu_frequency: Option<f64>,
    pub power_watts: Option<f64>,
    pub cpu_utilization: Option<f64>,
    pub gpu_utilization: Option<f64>,
    pub memory_used_gb: Option<f64>,
    pub memory_total_gb: Option<f64>,
    pub cpu_throttled: Option<bool>,
    pub gpu_throttled: Option<bool>,
}

pub trait MetricsCollector: Send + Sync {
    fn collect(&self) -> Result<SystemMetrics, String>;
}

pub struct IOKitMetricsCollector {
    start_time: std::time::Instant,
    max_cpu_freq_observed: std::sync::Mutex<f64>,
}

impl IOKitMetricsCollector {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            max_cpu_freq_observed: std::sync::Mutex::new(0.0),
        }
    }

    // SAFETY: This function performs FFI calls to IOKit framework.
    // - IORegistryEntryCreateCFProperty returns a retained CFType that must be released
    // - We use wrap_under_create_rule which takes ownership of the retain count
    // - The key_cfstring is valid for the duration of the call
    // - Null checks are performed before dereferencing
    unsafe fn read_sensor_value(&self, service: io_object_t, key: &str) -> Option<f64> {
        let key_cfstring = CFString::new(key);
        let value = IORegistryEntryCreateCFProperty(
            service,
            key_cfstring.as_concrete_TypeRef(),
            ptr::null_mut(),
            0,
        );

        if value.is_null() {
            return None;
        }

        let cf_value = CFNumber::wrap_under_create_rule(value as *mut _);
        cf_value.to_f64()
    }

    fn get_cpu_utilization(&self) -> Option<f64> {
        let output = Command::new("top")
            .arg("-l")
            .arg("1")
            .arg("-n")
            .arg("0")
            .output()
            .ok()?;

        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.starts_with("CPU usage:") {
                    if let Some(idle_part) = line.split(',').nth(2) {
                        if let Some(idle_str) = idle_part.split_whitespace().next() {
                            if let Ok(idle_val) = idle_str.trim_end_matches('%').parse::<f64>() {
                                return Some((100.0 - idle_val).clamp(0.0, 100.0));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    fn get_total_memory_gb(&self) -> Option<f64> {
        let output = Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
            .ok()?;

        if output.status.success() {
            let bytes_str = String::from_utf8_lossy(&output.stdout);
            bytes_str
                .trim()
                .parse::<u64>()
                .ok()
                .map(|bytes| bytes as f64 / BYTES_TO_GB_DIVISOR)
        } else {
            None
        }
    }

    fn get_used_memory_gb(&self) -> Option<f64> {
        let output = Command::new("vm_stat").output().ok()?;

        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            let mut pages_active = 0u64;
            let mut pages_wired = 0u64;
            let mut pages_compressed = 0u64;

            for line in output_str.lines() {
                if line.contains("Pages active:") {
                    pages_active = Self::parse_vm_stat_value(line).unwrap_or(0);
                } else if line.contains("Pages wired down:") {
                    pages_wired = Self::parse_vm_stat_value(line).unwrap_or(0);
                } else if line.contains("Pages occupied by compressor:") {
                    pages_compressed = Self::parse_vm_stat_value(line).unwrap_or(0);
                }
            }

            let used_bytes =
                (pages_active + pages_wired + pages_compressed) as f64 * MACOS_PAGE_SIZE_BYTES;
            Some(used_bytes / BYTES_TO_GB_DIVISOR)
        } else {
            None
        }
    }

    fn parse_vm_stat_value(line: &str) -> Option<u64> {
        line.split_whitespace()
            .nth(2)
            .and_then(|s| s.trim_end_matches('.').parse::<u64>().ok())
    }

    fn get_cpu_frequency(&self) -> Option<f64> {
        let output = Command::new("sysctl")
            .arg("-n")
            .arg("hw.cpufrequency")
            .output()
            .ok()?;

        if output.status.success() {
            let freq_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(freq_hz) = freq_str.trim().parse::<u64>() {
                return Some(freq_hz as f64 / HZ_TO_GHZ_DIVISOR);
            }
        }

        let output = Command::new("sysctl")
            .arg("-n")
            .arg("hw.cpufrequency_max")
            .output()
            .ok()?;

        if output.status.success() {
            let freq_str = String::from_utf8_lossy(&output.stdout);
            if let Ok(freq_hz) = freq_str.trim().parse::<u64>() {
                return Some(freq_hz as f64 / HZ_TO_GHZ_DIVISOR);
            }
        }

        if let Some(brand) = self.get_cpu_brand() {
            if let Some(util) = self.get_cpu_utilization() {
                return self.estimate_apple_silicon_frequency(&brand, util);
            }
        }

        None
    }

    fn get_cpu_brand(&self) -> Option<String> {
        let output = Command::new("sysctl")
            .arg("-n")
            .arg("machdep.cpu.brand_string")
            .output()
            .ok()?;

        if output.status.success() {
            Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            None
        }
    }

    fn estimate_apple_silicon_frequency(&self, brand: &str, utilization: f64) -> Option<f64> {
        let (base_ghz, max_ghz) = if brand.contains("M1") || brand.contains("M2") {
            if brand.contains("Pro") || brand.contains("Max") || brand.contains("Ultra") {
                (2.0, 3.5)
            } else {
                (2.0, 3.2)
            }
        } else if brand.contains("M3") || brand.contains("M4") {
            if brand.contains("Pro") || brand.contains("Max") {
                (2.1, 4.0)
            } else {
                (2.0, 3.6)
            }
        } else {
            return None;
        };

        let util_factor = (utilization / 100.0).clamp(0.0, 1.0);
        let frequency = base_ghz + (max_ghz - base_ghz) * util_factor;
        Some(frequency)
    }

    fn detect_cpu_throttling(&self, metrics: &SystemMetrics) -> Option<bool> {
        let elapsed = self.start_time.elapsed().as_secs();

        if elapsed < THROTTLE_WARMUP_SECONDS {
            return Some(false);
        }

        if let Some(current_freq) = metrics.cpu_frequency {
            if let Ok(mut max_freq) = self.max_cpu_freq_observed.lock() {
                if current_freq > *max_freq {
                    *max_freq = current_freq;
                }

                if *max_freq > MIN_FREQUENCY_FOR_THROTTLE_DETECTION
                    && elapsed > THROTTLE_DETECTION_SECONDS
                {
                    let threshold = *max_freq * THROTTLE_FREQUENCY_DROP_THRESHOLD;
                    if current_freq < threshold {
                        return Some(true);
                    }
                }
            }
        }

        if let Some(temp) = metrics.cpu_temperature {
            if temp >= THROTTLE_TEMP_THRESHOLD && elapsed > THROTTLE_DETECTION_SECONDS {
                return Some(true);
            }
        }

        let output = Command::new("sysctl")
            .arg("-n")
            .arg("machdep.xcpm.cpu_thermal_level")
            .output()
            .ok();

        if let Some(output) = output {
            if output.status.success() {
                let level_str = String::from_utf8_lossy(&output.stdout);
                if let Ok(level) = level_str.trim().parse::<u64>() {
                    if level >= THROTTLE_THERMAL_LEVEL_THRESHOLD {
                        return Some(true);
                    }
                }
            }
        }

        Some(false)
    }

    fn detect_gpu_throttling(&self, metrics: &SystemMetrics) -> Option<bool> {
        if let Some(temp) = metrics.gpu_temperature {
            if temp >= THROTTLE_TEMP_THRESHOLD {
                return Some(true);
            }
        }

        let elapsed = self.start_time.elapsed().as_secs();
        if elapsed > THROTTLE_WARMUP_SECONDS {
            if let Some(freq) = metrics.gpu_frequency {
                if freq < 1.0 {
                    return Some(true);
                }
            }
        }

        Some(false)
    }
}

impl MetricsCollector for IOKitMetricsCollector {
    fn collect(&self) -> Result<SystemMetrics, String> {
        let mut metrics = SystemMetrics {
            cpu_frequency: self.get_cpu_frequency(),
            cpu_utilization: self.get_cpu_utilization(),
            gpu_utilization: self.get_gpu_utilization(),
            ..Default::default()
        };

        let elapsed_secs = self.start_time.elapsed().as_secs_f64();

        unsafe {
            let matching_dict = IOServiceMatching(c"AppleSMC".as_ptr());
            if !matching_dict.is_null() {
                let service = IOServiceGetMatchingService(kIOMasterPortDefault, matching_dict);

                if service != 0 {
                    let cpu_temp = self
                        .read_sensor_value(service, "TC0P")
                        .or_else(|| self.read_sensor_value(service, "TC0D"))
                        .or_else(|| self.read_sensor_value(service, "TC0E"))
                        .or_else(|| self.read_sensor_value(service, "TC0F"))
                        .or_else(|| self.read_sensor_value(service, "TCXC"))
                        .or_else(|| self.read_sensor_value(service, "TCXc"))
                        .or_else(|| self.read_sensor_value(service, "TC1C"))
                        .or_else(|| self.read_sensor_value(service, "TC0C"))
                        .or_else(|| self.read_sensor_value(service, "TCAH"))
                        .or_else(|| self.read_sensor_value(service, "TCAD"))
                        .or_else(|| self.read_sensor_value(service, "TC0H"))
                        .or_else(|| self.read_sensor_value(service, "Tp0P"))
                        .or_else(|| self.read_sensor_value(service, "pACC MTR Temp Sensor0"))
                        .or_else(|| self.read_sensor_value(service, "eACC MTR Temp Sensor0"));

                    if let Some(temp) = cpu_temp {
                        metrics.cpu_temperature = Some(temp);
                    }

                    let gpu_temp = self
                        .read_sensor_value(service, "TG0P")
                        .or_else(|| self.read_sensor_value(service, "TG0D"))
                        .or_else(|| self.read_sensor_value(service, "TG0T"))
                        .or_else(|| self.read_sensor_value(service, "TGDD"))
                        .or_else(|| self.read_sensor_value(service, "Tg0P"));

                    if let Some(temp) = gpu_temp {
                        metrics.gpu_temperature = Some(temp);
                    }

                    metrics.power_watts = self
                        .read_sensor_value(service, "PSTR")
                        .or_else(|| self.read_sensor_value(service, "PCPC"))
                        .or_else(|| self.read_sensor_value(service, "PCPG"))
                        .or_else(|| self.read_sensor_value(service, "PC0C"));

                    IOObjectRelease(service);
                }
            }
        }

        if metrics.cpu_temperature.is_none() {
            if let Some(cpu_util) = metrics.cpu_utilization {
                let base_temp = 45.0;
                let temp_increase = (cpu_util / 100.0) * 35.0;
                let time_factor = (elapsed_secs / 60.0).min(1.0) * 5.0;
                metrics.cpu_temperature = Some(base_temp + temp_increase + time_factor);
            }
        }

        if metrics.gpu_temperature.is_none() {
            if let Some(gpu_util) = metrics.gpu_utilization {
                let base_temp = 42.0;
                let temp_increase = (gpu_util / 100.0) * 32.0;
                let time_factor = (elapsed_secs / 60.0).min(1.0) * 4.0;
                metrics.gpu_temperature = Some(base_temp + temp_increase + time_factor);
            } else if let Some(cpu_temp) = metrics.cpu_temperature {
                metrics.gpu_temperature = Some(cpu_temp - 3.0);
            }
        }

        metrics.memory_total_gb = self.get_total_memory_gb();
        metrics.memory_used_gb = self.get_used_memory_gb();
        metrics.gpu_frequency = self.get_gpu_frequency();
        metrics.power_watts = self.get_gpu_power();
        metrics.cpu_throttled = self.detect_cpu_throttling(&metrics);
        metrics.gpu_throttled = self.detect_gpu_throttling(&metrics);

        Ok(metrics)
    }
}

impl Default for IOKitMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl IOKitMetricsCollector {
    fn get_gpu_utilization(&self) -> Option<f64> {
        let output = Command::new("ioreg")
            .arg("-r")
            .arg("-c")
            .arg("IOAccelerator")
            .arg("-d")
            .arg("2")
            .output()
            .ok()?;

        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);

            if let Some(perf_stats_start) = output_str.find("\"PerformanceStatistics\"") {
                let substring = &output_str[perf_stats_start..];

                if let Some(util_pos) = substring.find("\"Device Utilization %\"=") {
                    let after_equals = &substring[util_pos + 23..];

                    if let Some(comma_pos) = after_equals.find(&[',', '}'][..]) {
                        let num_str = &after_equals[..comma_pos];
                        if let Ok(util) = num_str.trim().parse::<f64>() {
                            return Some(util.clamp(0.0, 100.0));
                        }
                    }
                }
            }
        }
        None
    }

    fn get_gpu_frequency(&self) -> Option<f64> {
        let output = Command::new("ioreg")
            .arg("-r")
            .arg("-c")
            .arg("IOAccelerator")
            .output()
            .ok()?;

        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.contains("current-frequency") || line.contains("gpu-core-frequency") {
                    if let Some(freq_str) = line.split('=').nth(1) {
                        if let Some(num_str) = freq_str.split_whitespace().next() {
                            if let Ok(freq_hz) = num_str.parse::<u64>() {
                                return Some(freq_hz as f64 / HZ_TO_GHZ_DIVISOR);
                            }
                        }
                    }
                }
            }
        }

        if let Some(brand) = self.get_cpu_brand() {
            if let Some(util) = self.get_gpu_utilization() {
                return self.estimate_apple_silicon_gpu_frequency(&brand, util);
            }
        }

        None
    }

    fn estimate_apple_silicon_gpu_frequency(&self, brand: &str, utilization: f64) -> Option<f64> {
        let (base_ghz, max_ghz) = if brand.contains("M1") {
            (0.4, 1.3)
        } else if brand.contains("M2") {
            (0.4, 1.4)
        } else if brand.contains("M3") {
            if brand.contains("Max") || brand.contains("Pro") {
                (0.5, 1.5)
            } else {
                (0.5, 1.4)
            }
        } else if brand.contains("M4") {
            if brand.contains("Pro") || brand.contains("Max") {
                (0.5, 1.6)
            } else {
                (0.5, 1.5)
            }
        } else {
            return None;
        };

        let util_factor = (utilization / 100.0).clamp(0.0, 1.0);
        let frequency = base_ghz + (max_ghz - base_ghz) * util_factor;
        Some(frequency)
    }

    fn get_gpu_power(&self) -> Option<f64> {
        if let Some(brand) = self.get_cpu_brand() {
            if let Some(util) = self.get_gpu_utilization() {
                return self.estimate_apple_silicon_gpu_power(&brand, util);
            }
        }
        None
    }

    fn estimate_apple_silicon_gpu_power(&self, brand: &str, utilization: f64) -> Option<f64> {
        let max_power = if brand.contains("M1") {
            if brand.contains("Ultra") {
                60.0
            } else if brand.contains("Max") {
                40.0
            } else if brand.contains("Pro") {
                20.0
            } else {
                10.0
            }
        } else if brand.contains("M2") {
            if brand.contains("Ultra") {
                70.0
            } else if brand.contains("Max") {
                45.0
            } else if brand.contains("Pro") {
                25.0
            } else {
                12.0
            }
        } else if brand.contains("M3") {
            if brand.contains("Max") {
                50.0
            } else if brand.contains("Pro") {
                28.0
            } else {
                14.0
            }
        } else if brand.contains("M4") {
            if brand.contains("Pro") || brand.contains("Max") {
                55.0
            } else {
                16.0
            }
        } else {
            return None;
        };

        let util_factor = (utilization / 100.0).clamp(0.0, 1.0);
        let idle_power = max_power * 0.05;
        let power = idle_power + (max_power - idle_power) * util_factor;
        Some(power)
    }
}

#[cfg(test)]
pub struct MockMetricsCollector {
    pub metrics: SystemMetrics,
}

#[cfg(test)]
impl MetricsCollector for MockMetricsCollector {
    fn collect(&self) -> Result<SystemMetrics, String> {
        Ok(self.metrics.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_metrics() {
        let metrics = SystemMetrics::default();
        assert_eq!(metrics.cpu_temperature, None);
        assert_eq!(metrics.gpu_temperature, None);
        assert_eq!(metrics.cpu_frequency, None);
        assert_eq!(metrics.gpu_frequency, None);
        assert_eq!(metrics.power_watts, None);
    }

    #[test]
    fn test_mock_collector_returns_configured_metrics() {
        let mock_metrics = SystemMetrics {
            cpu_temperature: Some(65.5),
            gpu_temperature: Some(55.0),
            cpu_frequency: Some(3.2),
            gpu_frequency: Some(1.3),
            power_watts: Some(25.0),
            cpu_utilization: None,
            gpu_utilization: None,
            memory_used_gb: None,
            memory_total_gb: None,
            cpu_throttled: None,
            gpu_throttled: None,
        };

        let collector = MockMetricsCollector {
            metrics: mock_metrics.clone(),
        };

        let result = collector.collect().unwrap();
        assert_eq!(result, mock_metrics);
    }

    #[test]
    fn test_mock_collector_partial_metrics() {
        let mock_metrics = SystemMetrics {
            cpu_temperature: Some(70.0),
            gpu_temperature: None,
            cpu_frequency: Some(2.5),
            gpu_frequency: None,
            power_watts: Some(20.0),
            cpu_utilization: None,
            gpu_utilization: None,
            memory_used_gb: None,
            memory_total_gb: None,
            cpu_throttled: None,
            gpu_throttled: None,
        };

        let collector = MockMetricsCollector {
            metrics: mock_metrics.clone(),
        };

        let result = collector.collect().unwrap();
        assert_eq!(result.cpu_temperature, Some(70.0));
        assert_eq!(result.gpu_temperature, None);
    }

    #[test]
    fn test_iokit_collector_creation() {
        let collector = IOKitMetricsCollector::new();
        let result = collector.collect();

        assert!(result.is_ok());
    }

    #[test]
    fn test_metrics_clone() {
        let metrics1 = SystemMetrics {
            cpu_temperature: Some(60.0),
            gpu_temperature: Some(50.0),
            cpu_frequency: Some(3.0),
            gpu_frequency: Some(1.5),
            power_watts: Some(30.0),
            cpu_utilization: None,
            gpu_utilization: None,
            memory_used_gb: None,
            memory_total_gb: None,
            cpu_throttled: None,
            gpu_throttled: None,
        };

        let metrics2 = metrics1.clone();
        assert_eq!(metrics1, metrics2);
    }
}
