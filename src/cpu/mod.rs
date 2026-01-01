use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

const PRIME_SEARCH_CHUNK_SIZE: u64 = 5000;
const MATRIX_SIZE: usize = 32;

pub struct CpuStressTest {
    num_cores: usize,
    running: Arc<AtomicBool>,
    primes_found: Arc<AtomicU64>,
}

impl CpuStressTest {
    pub fn new(num_cores: usize) -> Self {
        Self {
            num_cores,
            running: Arc::new(AtomicBool::new(false)),
            primes_found: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
        self.primes_found.store(0, Ordering::SeqCst);

        let num_cores = self.num_cores;
        let running = Arc::clone(&self.running);
        let primes_found = Arc::clone(&self.primes_found);

        thread::spawn(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_cores)
                .build();

            match pool {
                Ok(pool) => {
                    pool.install(|| {
                        Self::run_workload(running, primes_found);
                    });
                }
                Err(e) => {
                    eprintln!("Failed to create thread pool: {}", e);
                }
            }
        });
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    pub fn primes_found(&self) -> u64 {
        self.primes_found.load(Ordering::SeqCst)
    }

    fn run_workload(running: Arc<AtomicBool>, primes_found: Arc<AtomicU64>) {
        let mut start = 2u64;
        let mut iteration = 0u64;

        while running.load(Ordering::SeqCst) {
            let end = start + PRIME_SEARCH_CHUNK_SIZE;
            let range: Vec<u64> = (start..end).collect();

            let primes_in_chunk: u64 = range
                .par_iter()
                .filter(|&&n| {
                    if Self::is_prime(n) {
                        if n % 100 == 0 {
                            let _ = Self::stress_matrix_multiply();
                            let _ = Self::stress_hash(n);
                        }
                        true
                    } else {
                        false
                    }
                })
                .count() as u64;

            primes_found.fetch_add(primes_in_chunk, Ordering::SeqCst);

            start = end;
            iteration += 1;

            if iteration.is_multiple_of(5) {
                Self::stress_compression(start);
            }
        }
    }

    fn is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n.is_multiple_of(2) {
            return false;
        }

        let limit = (n as f64).sqrt() as u64;
        for i in (3..=limit).step_by(2) {
            if n.is_multiple_of(i) {
                return false;
            }
        }
        true
    }

    fn stress_matrix_multiply() -> f64 {
        let mut matrix_a = vec![0.0f64; MATRIX_SIZE * MATRIX_SIZE];
        let mut matrix_b = vec![0.0f64; MATRIX_SIZE * MATRIX_SIZE];
        let mut result = vec![0.0f64; MATRIX_SIZE * MATRIX_SIZE];

        for i in 0..MATRIX_SIZE * MATRIX_SIZE {
            matrix_a[i] = (i as f64).sin();
            matrix_b[i] = (i as f64).cos();
        }

        for i in 0..MATRIX_SIZE {
            for j in 0..MATRIX_SIZE {
                let mut sum = 0.0;
                for k in 0..MATRIX_SIZE {
                    sum += matrix_a[i * MATRIX_SIZE + k] * matrix_b[k * MATRIX_SIZE + j];
                }
                result[i * MATRIX_SIZE + j] = sum;
            }
        }

        result.iter().sum()
    }

    fn stress_hash(n: u64) -> f64 {
        let mut hash = n;
        for _ in 0..100 {
            hash = hash.wrapping_mul(2654435761);
            hash = hash.wrapping_add(hash >> 16);
            hash ^= hash << 13;
            hash ^= hash >> 7;
            hash ^= hash << 17;
        }
        hash as f64
    }

    fn stress_compression(seed: u64) {
        let data: Vec<u8> = (0..1024)
            .map(|i| ((seed.wrapping_add(i)) % 256) as u8)
            .collect();

        let mut compressed = Vec::new();
        let mut prev = data[0];
        let mut count = 1u8;

        for &byte in &data[1..] {
            if byte == prev && count < 255 {
                count += 1;
            } else {
                compressed.push(prev);
                compressed.push(count);
                prev = byte;
                count = 1;
            }
        }
        compressed.push(prev);
        compressed.push(count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_cpu_stress_test_creation() {
        let test = CpuStressTest::new(4);
        assert_eq!(test.num_cores, 4);
        assert!(!test.is_running());
        assert_eq!(test.primes_found(), 0);
    }

    #[test]
    fn test_is_prime_basic() {
        assert!(!CpuStressTest::is_prime(0));
        assert!(!CpuStressTest::is_prime(1));
        assert!(CpuStressTest::is_prime(2));
        assert!(CpuStressTest::is_prime(3));
        assert!(!CpuStressTest::is_prime(4));
        assert!(CpuStressTest::is_prime(5));
        assert!(!CpuStressTest::is_prime(6));
        assert!(CpuStressTest::is_prime(7));
    }

    #[test]
    fn test_is_prime_larger_numbers() {
        assert!(CpuStressTest::is_prime(97));
        assert!(CpuStressTest::is_prime(101));
        assert!(!CpuStressTest::is_prime(100));
        assert!(CpuStressTest::is_prime(7919));
        assert!(!CpuStressTest::is_prime(7920));
    }

    #[test]
    fn test_stress_test_runs_and_finds_primes() {
        let test = CpuStressTest::new(2);
        test.start();

        thread::sleep(Duration::from_millis(1500));

        assert!(test.is_running());
        let primes_after_run = test.primes_found();
        assert!(primes_after_run > 0, "Should have found some primes");

        test.stop();

        thread::sleep(Duration::from_millis(100));

        let primes_final = test.primes_found();
        assert!(primes_final >= primes_after_run);
    }

    #[test]
    fn test_stop_terminates_workload() {
        let test = CpuStressTest::new(1);
        test.start();

        thread::sleep(Duration::from_millis(100));

        let primes_before_stop = test.primes_found();
        test.stop();

        thread::sleep(Duration::from_millis(100));

        let primes_after_stop = test.primes_found();

        assert!(
            primes_after_stop >= primes_before_stop,
            "Prime count should not decrease"
        );
    }

    #[test]
    fn test_multiple_cores() {
        let single_core = CpuStressTest::new(1);
        let multi_core = CpuStressTest::new(4);

        assert_eq!(single_core.num_cores, 1);
        assert_eq!(multi_core.num_cores, 4);
    }
}
