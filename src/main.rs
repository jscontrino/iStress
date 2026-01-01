use clap::Parser;
use istress::cpu::CpuStressTest;
use istress::gpu::GpuStressTest;
use istress::metrics::{IOKitMetricsCollector, MetricsCollector};
use istress::ui::{run_ui, StressTestUI};
use istress::Cli;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let cli = Cli::parse();

    if let Err(e) = cli.validate() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }

    println!("Starting iStress - Apple Silicon Stress Test");
    println!("Configuration:");
    println!("  CPU: {}", cli.cpu);
    println!("  GPU: {}", cli.gpu);
    if let Some(cores) = cli.cores {
        println!("  Cores: {}", cores);
    }
    if let Some(duration) = &cli.duration {
        println!("  Duration: {}", duration);
    }
    println!();

    if let Err(e) = run_stress_test(&cli) {
        eprintln!("Error running stress test: {}", e);
        std::process::exit(1);
    }
}

fn run_stress_test(cli: &Cli) -> Result<(), String> {
    let metrics_collector = Arc::new(IOKitMetricsCollector::new());

    let cpu_test = if cli.cpu {
        let cores = cli.get_core_count();
        println!("Initializing CPU stress test with {} cores...", cores);
        Some(CpuStressTest::new(cores))
    } else {
        None
    };

    let gpu_test = if cli.gpu {
        println!("Initializing GPU stress test...");
        match GpuStressTest::new() {
            Ok(test) => Some(test),
            Err(e) => {
                eprintln!("Warning: Failed to initialize GPU stress test: {}", e);
                None
            }
        }
    } else {
        None
    };

    if let Some(ref test) = cpu_test {
        test.start();
        println!("CPU stress test started");
    }

    if let Some(ref test) = gpu_test {
        if let Err(e) = test.start() {
            eprintln!("Warning: Failed to start GPU stress test: {}", e);
        } else {
            println!("GPU stress test started");
        }
    }

    println!("\nLaunching TUI...\n");

    let start_time = Instant::now();
    let duration_seconds = cli.get_duration_seconds();

    let ui = StressTestUI::new(cli.cpu, cli.gpu, duration_seconds);

    let ui_result = run_ui(ui, |ui| {
        if let Some(duration) = duration_seconds {
            if start_time.elapsed().as_secs() >= duration {
                return false;
            }
        }

        if let Ok(metrics) = metrics_collector.collect() {
            ui.update_metrics(metrics);
        }

        let cpu_primes = cpu_test.as_ref().map(|t| t.primes_found()).unwrap_or(0);
        let gpu_iterations = gpu_test.as_ref().map(|t| t.iterations()).unwrap_or(0);
        ui.update_workload_stats(cpu_primes, gpu_iterations);

        true
    });

    let actual_duration = start_time.elapsed().as_secs_f64();

    if let Some(ref test) = cpu_test {
        test.stop();
        println!("CPU stress test stopped");
    }

    if let Some(ref test) = gpu_test {
        test.stop();
        println!("GPU stress test stopped");
    }

    let ui = ui_result.map_err(|e| format!("UI error: {}", e))?;

    println!("\nStress test completed successfully!");

    if cli.report {
        let report = ui.generate_report(actual_duration);
        println!("{}", report);
    }

    Ok(())
}
