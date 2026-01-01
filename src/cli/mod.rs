use clap::Parser;

#[derive(Parser, Debug, Clone, PartialEq)]
#[command(name = "istress")]
#[command(about = "Stress test Apple Silicon MacBooks CPU and GPU", long_about = None)]
pub struct Cli {
    #[arg(long, help = "Enable CPU stress test")]
    pub cpu: bool,

    #[arg(long, help = "Enable GPU stress test")]
    pub gpu: bool,

    #[arg(long, help = "Number of CPU cores to use (default: all)")]
    pub cores: Option<usize>,

    #[arg(long, help = "Duration of stress test (e.g., 60s, 5m, 1h)")]
    pub duration: Option<String>,

    #[arg(long, help = "Print a summary report after the test completes")]
    pub report: bool,
}

impl Cli {
    pub fn validate(&self) -> Result<(), String> {
        if !self.cpu && !self.gpu {
            return Err("At least one of --cpu or --gpu must be specified".to_string());
        }

        if let Some(cores) = self.cores {
            if cores == 0 {
                return Err("Core count must be at least 1".to_string());
            }
        }

        if let Some(duration) = &self.duration {
            parse_duration(duration)?;
        }

        Ok(())
    }

    pub fn get_duration_seconds(&self) -> Option<u64> {
        self.duration.as_ref().and_then(|d| parse_duration(d).ok())
    }

    pub fn get_core_count(&self) -> usize {
        self.cores.unwrap_or_else(num_cpus)
    }
}

fn parse_duration(duration: &str) -> Result<u64, String> {
    let duration = duration.trim();

    if duration.is_empty() {
        return Err("Duration cannot be empty".to_string());
    }

    let (num_part, unit_part) = if let Some(pos) = duration.find(|c: char| c.is_alphabetic()) {
        (&duration[..pos], &duration[pos..])
    } else {
        return Err("Duration must include a unit (s, m, h)".to_string());
    };

    let num: u64 = num_part
        .parse()
        .map_err(|_| format!("Invalid number in duration: {}", num_part))?;

    let multiplier = match unit_part.to_lowercase().as_str() {
        "s" | "sec" | "secs" | "second" | "seconds" => 1,
        "m" | "min" | "mins" | "minute" | "minutes" => 60,
        "h" | "hr" | "hrs" | "hour" | "hours" => 3600,
        _ => return Err(format!("Unknown duration unit: {}", unit_part)),
    };

    Ok(num * multiplier)
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_requires_at_least_one_flag() {
        let cli = Cli {
            cpu: false,
            gpu: false,
            cores: None,
            duration: None,
            report: false,
        };
        assert!(cli.validate().is_err());
    }

    #[test]
    fn test_cli_cpu_only() {
        let cli = Cli {
            cpu: true,
            gpu: false,
            cores: None,
            duration: None,
            report: false,
        };
        assert!(cli.validate().is_ok());
    }

    #[test]
    fn test_cli_gpu_only() {
        let cli = Cli {
            cpu: false,
            gpu: true,
            cores: None,
            duration: None,
            report: false,
        };
        assert!(cli.validate().is_ok());
    }

    #[test]
    fn test_cli_both_cpu_and_gpu() {
        let cli = Cli {
            cpu: true,
            gpu: true,
            cores: Some(4),
            duration: Some("60s".to_string()),
            report: false,
        };
        assert!(cli.validate().is_ok());
    }

    #[test]
    fn test_cli_zero_cores_invalid() {
        let cli = Cli {
            cpu: true,
            gpu: false,
            cores: Some(0),
            duration: None,
            report: false,
        };
        assert!(cli.validate().is_err());
    }

    #[test]
    fn test_parse_duration_seconds() {
        assert_eq!(parse_duration("30s").unwrap(), 30);
        assert_eq!(parse_duration("45sec").unwrap(), 45);
        assert_eq!(parse_duration("60seconds").unwrap(), 60);
    }

    #[test]
    fn test_parse_duration_minutes() {
        assert_eq!(parse_duration("5m").unwrap(), 300);
        assert_eq!(parse_duration("10min").unwrap(), 600);
        assert_eq!(parse_duration("2minutes").unwrap(), 120);
    }

    #[test]
    fn test_parse_duration_hours() {
        assert_eq!(parse_duration("1h").unwrap(), 3600);
        assert_eq!(parse_duration("2hr").unwrap(), 7200);
        assert_eq!(parse_duration("3hours").unwrap(), 10800);
    }

    #[test]
    fn test_parse_duration_invalid_unit() {
        assert!(parse_duration("30x").is_err());
        assert!(parse_duration("45days").is_err());
    }

    #[test]
    fn test_parse_duration_invalid_number() {
        assert!(parse_duration("abcs").is_err());
        assert!(parse_duration("s").is_err());
    }

    #[test]
    fn test_parse_duration_no_unit() {
        assert!(parse_duration("30").is_err());
    }

    #[test]
    fn test_get_duration_seconds() {
        let cli = Cli {
            cpu: true,
            gpu: false,
            cores: None,
            duration: Some("2m".to_string()),
            report: false,
        };
        assert_eq!(cli.get_duration_seconds(), Some(120));
    }

    #[test]
    fn test_get_duration_seconds_none() {
        let cli = Cli {
            cpu: true,
            gpu: false,
            cores: None,
            duration: None,
            report: false,
        };
        assert_eq!(cli.get_duration_seconds(), None);
    }

    #[test]
    fn test_get_core_count_default() {
        let cli = Cli {
            cpu: true,
            gpu: false,
            cores: None,
            duration: None,
            report: false,
        };
        assert!(cli.get_core_count() > 0);
    }

    #[test]
    fn test_get_core_count_specified() {
        let cli = Cli {
            cpu: true,
            gpu: false,
            cores: Some(4),
            duration: None,
            report: false,
        };
        assert_eq!(cli.get_core_count(), 4);
    }
}
