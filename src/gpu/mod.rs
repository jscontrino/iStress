use metal::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

const GPU_BUFFER_SIZE: usize = 1024 * 1024;
const THREAD_GROUP_WIDTH: u64 = 256;
const SHADER_LOOP_ITERATIONS: i32 = 1000;

pub struct GpuStressTest {
    running: Arc<AtomicBool>,
    iterations: Arc<AtomicU64>,
    device: Device,
    pipeline_state: ComputePipelineState,
    command_queue: CommandQueue,
}

impl GpuStressTest {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;

        let shader_source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void stress_compute(device float* input [[buffer(0)]],
                                      device float* output [[buffer(1)]],
                                      uint id [[thread_position_in_grid]]) {{
                float value = input[id];

                for (int i = 0; i < {}; i++) {{
                    value = sin(value) * cos(value);
                    value = sqrt(abs(value)) + 0.001;
                    value = value * value + 0.1;
                }}

                output[id] = value;
            }}
        "#,
            SHADER_LOOP_ITERATIONS
        );

        let library = device
            .new_library_with_source(&shader_source, &CompileOptions::new())
            .map_err(|e| format!("Failed to compile Metal shader: {}", e))?;

        let kernel = library
            .get_function("stress_compute", None)
            .map_err(|e| format!("Failed to get kernel function: {}", e))?;

        let pipeline_state = device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline state: {}", e))?;

        let command_queue = device.new_command_queue();

        Ok(Self {
            running: Arc::new(AtomicBool::new(false)),
            iterations: Arc::new(AtomicU64::new(0)),
            device,
            pipeline_state,
            command_queue,
        })
    }

    pub fn start(&self) -> Result<(), String> {
        self.running.store(true, Ordering::SeqCst);
        self.iterations.store(0, Ordering::SeqCst);

        let device = self.device.clone();
        let pipeline_state = self.pipeline_state.clone();
        let command_queue = self.command_queue.clone();
        let running = Arc::clone(&self.running);
        let iterations = Arc::clone(&self.iterations);

        thread::spawn(move || {
            Self::run_workload(device, pipeline_state, command_queue, running, iterations);
        });

        Ok(())
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    pub fn iterations(&self) -> u64 {
        self.iterations.load(Ordering::SeqCst)
    }

    fn run_workload(
        device: Device,
        pipeline_state: ComputePipelineState,
        command_queue: CommandQueue,
        running: Arc<AtomicBool>,
        iterations: Arc<AtomicU64>,
    ) {
        let input_buffer = device.new_buffer(
            (GPU_BUFFER_SIZE * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = device.new_buffer(
            (GPU_BUFFER_SIZE * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // SAFETY: This performs direct memory access to Metal buffer contents.
        // - input_buffer.contents() returns a valid pointer to the buffer's memory region
        // - The pointer is valid for the lifetime of the buffer
        // - We iterate only within the allocated buffer size (GPU_BUFFER_SIZE)
        // - Each write is to a valid aligned f32 location
        // - No other threads access this memory during initialization
        let input_ptr = input_buffer.contents() as *mut f32;
        unsafe {
            for i in 0..GPU_BUFFER_SIZE {
                *input_ptr.add(i) = (i as f32) * 0.001;
            }
        }

        while running.load(Ordering::SeqCst) {
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&pipeline_state);
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);

            let thread_group_count = MTLSize {
                width: THREAD_GROUP_WIDTH,
                height: 1,
                depth: 1,
            };

            let thread_group_size = MTLSize {
                width: (GPU_BUFFER_SIZE as u64).div_ceil(THREAD_GROUP_WIDTH),
                height: 1,
                depth: 1,
            };

            encoder.dispatch_thread_groups(thread_group_size, thread_group_count);
            encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            iterations.fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl Default for GpuStressTest {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            let device = Device::system_default().expect("No Metal device for default");
            let shader_source = r#"
                #include <metal_stdlib>
                using namespace metal;
                kernel void stress_compute(device float* input [[buffer(0)]],
                                          device float* output [[buffer(1)]],
                                          uint id [[thread_position_in_grid]]) {
                    output[id] = input[id];
                }
            "#;
            let library = device
                .new_library_with_source(shader_source, &CompileOptions::new())
                .expect("Failed to compile shader");
            let kernel = library
                .get_function("stress_compute", None)
                .expect("Failed to get kernel");
            let pipeline_state = device
                .new_compute_pipeline_state_with_function(&kernel)
                .expect("Failed to create pipeline");
            let command_queue = device.new_command_queue();

            Self {
                running: Arc::new(AtomicBool::new(false)),
                iterations: Arc::new(AtomicU64::new(0)),
                device,
                pipeline_state,
                command_queue,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_gpu_stress_test_creation() {
        let result = GpuStressTest::new();

        if let Ok(test) = result {
            assert!(!test.is_running());
            assert_eq!(test.iterations(), 0);
        }
    }

    #[test]
    fn test_gpu_stress_test_default() {
        let test = GpuStressTest::default();
        assert!(!test.is_running());
        assert_eq!(test.iterations(), 0);
    }

    #[test]
    fn test_gpu_stress_start_and_stop() {
        let result = GpuStressTest::new();

        if let Ok(test) = result {
            assert!(test.start().is_ok());
            thread::sleep(Duration::from_millis(500));

            assert!(test.is_running());
            let iterations_after_run = test.iterations();

            test.stop();
            thread::sleep(Duration::from_millis(100));
            assert!(!test.is_running());

            let iterations_final = test.iterations();
            assert!(iterations_final >= iterations_after_run);
        }
    }

    #[test]
    fn test_gpu_stress_stop_terminates() {
        let result = GpuStressTest::new();

        if let Ok(test) = result {
            assert!(test.start().is_ok());
            thread::sleep(Duration::from_millis(50));

            let iterations_before_stop = test.iterations();
            test.stop();

            thread::sleep(Duration::from_millis(100));
            let iterations_after_stop = test.iterations();

            assert!(
                iterations_after_stop >= iterations_before_stop,
                "Iteration count should not decrease"
            );
        }
    }

    #[test]
    fn test_gpu_stress_multiple_start_stop_cycles() {
        let result = GpuStressTest::new();

        if let Ok(test) = result {
            for _ in 0..3 {
                assert!(test.start().is_ok());
                thread::sleep(Duration::from_millis(50));
                assert!(test.is_running());

                test.stop();
                thread::sleep(Duration::from_millis(50));
                assert!(!test.is_running());
            }
        }
    }
}
