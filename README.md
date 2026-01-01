# iStress

A high-performance CLI tool for stress testing Apple Silicon MacBook CPUs and GPUs, built with Rust using test-driven development.

## Features

- **CPU Stress Testing**: Multi-threaded prime number calculation workload
- **GPU Stress Testing**: Metal compute shaders for intensive GPU workload
- **Real-time Metrics**: Live monitoring of temperature, clock speed, and power consumption
- **Interactive TUI**: Beautiful terminal interface with smooth line graphs showing temperature trends
- **Visual Graphs**: Bold, connected line graphs with labeled X/Y axes for easy reading
- **Auto-scaling**: Y-axis automatically scales to data range for optimal visibility
- **Configurable**: Control core count, test duration, and workload types via CLI flags
- **Fully Tested**: Built with TDD, 38 passing tests

## Requirements

- macOS with Apple Silicon (M1, M2, M3, etc.)
- Rust toolchain (latest stable)

## Installation

```bash
cargo build --release
```

The binary will be available at `./target/release/istress`

## Usage

### Basic Examples

Run CPU stress test:
```bash
istress --cpu
```

Run GPU stress test:
```bash
istress --gpu
```

Run both CPU and GPU stress tests:
```bash
istress --cpu --gpu
```

### Advanced Options

Specify number of CPU cores:
```bash
istress --cpu --cores 4
```

Set test duration:
```bash
istress --cpu --duration 60s    # 60 seconds
istress --cpu --duration 5m     # 5 minutes
istress --cpu --duration 1h     # 1 hour
```

Combined example:
```bash
istress --cpu --gpu --cores 8 --duration 2m
```

Generate a final report:
```bash
istress --cpu --gpu --duration 1m --report
```

### CLI Flags

- `--cpu`: Enable CPU stress test
- `--gpu`: Enable GPU stress test
- `--cores <N>`: Number of CPU cores to use (default: all available)
- `--duration <TIME>`: Duration of stress test (e.g., 60s, 5m, 1h)
- `--report`: Print a summary report after the test completes
- `--help`: Show help information

### Interactive Controls

- Press `q` to quit the stress test

## Architecture

The project is organized into modular components:

- **CLI Module** (`src/cli/mod.rs`): Command-line argument parsing with clap
- **CPU Module** (`src/cpu/mod.rs`): CPU stress testing with prime number calculations
- **GPU Module** (`src/gpu/mod.rs`): GPU stress testing with Metal compute shaders
- **Metrics Module** (`src/metrics/mod.rs`): System metrics collection via IOKit FFI
- **UI Module** (`src/ui/mod.rs`): Terminal UI with ratatui for real-time visualization
- **Main** (`src/main.rs`): Orchestration and coordination of all components

## Development

### Running Tests

Run all tests:
```bash
cargo test
```

Run tests for a specific module:
```bash
cargo test --lib cli::tests
cargo test --lib cpu::tests
cargo test --lib gpu::tests
cargo test --lib metrics::tests
cargo test --lib ui::tests
```

### Building

Debug build:
```bash
cargo build
```

Release build (optimized):
```bash
cargo build --release
```

## Technical Details

### CPU Stress Test
- Uses rayon for parallel processing
- Implements efficient prime number calculation algorithm
- Configurable core count
- Tracks total primes found

### GPU Stress Test
- Metal compute shaders with intensive math operations
- 1M float buffer processing
- Trigonometric and arithmetic operations per iteration
- Tracks completed iterations

### Metrics Collection
- Direct IOKit framework integration via FFI
- Collects CPU/GPU temperature (peak, average, current)
- Collects CPU/GPU utilization percentage
- Monitors clock frequencies
- Tracks power consumption
- Tracks memory usage
- Detects thermal throttling
- Mock implementations for testing

### Final Report
- Optional summary report with `--report` flag
- Shows peak, average, and final temperatures
- Displays CPU/GPU utilization and frequencies
- Reports workload statistics (primes/second, iterations/second)
- Indicates thermal throttling events
- Shows memory usage statistics

### UI
- Real-time terminal interface with ratatui
- Live temperature graphs (CPU and GPU)
- Metrics display (temperature, frequency, power)
- Workload statistics (primes found, GPU iterations)
- 500ms refresh rate

## License

This project was created as a learning exercise in Rust, TDD, and systems programming.

## Contributing

This is a personal project, but feel free to fork and modify for your own use!

## Safety Notes

Stress testing can:
- Generate significant heat
- Consume substantial power
- Potentially reduce battery life during extended tests
- Cause thermal throttling on sustained loads

Use responsibly and monitor your system temperatures.
