use crate::metrics::SystemMetrics;
use chrono::Local;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph},
    Frame, Terminal,
};
use std::io;
use std::time::{Duration, Instant};

pub struct MetricsHistory {
    pub timestamps: Vec<f64>,
    pub cpu_temps: Vec<f64>,
    pub gpu_temps: Vec<f64>,
    pub cpu_freqs: Vec<f64>,
    pub power: Vec<f64>,
    pub cpu_utilizations: Vec<f64>,
    pub gpu_utilizations: Vec<f64>,
    pub memory_used: Vec<f64>,
    start_time: Instant,
    max_duration_secs: f64,
}

impl MetricsHistory {
    pub fn new(duration_secs: f64) -> Self {
        Self {
            timestamps: Vec::new(),
            cpu_temps: Vec::new(),
            gpu_temps: Vec::new(),
            cpu_freqs: Vec::new(),
            power: Vec::new(),
            cpu_utilizations: Vec::new(),
            gpu_utilizations: Vec::new(),
            memory_used: Vec::new(),
            start_time: Instant::now(),
            max_duration_secs: duration_secs,
        }
    }

    pub fn add_sample(&mut self, metrics: &SystemMetrics) {
        let elapsed = self.start_time.elapsed().as_secs_f64();

        self.timestamps.push(elapsed);

        if let Some(temp) = metrics.cpu_temperature {
            self.cpu_temps.push(temp);
        }

        if let Some(temp) = metrics.gpu_temperature {
            self.gpu_temps.push(temp);
        }

        if let Some(freq) = metrics.cpu_frequency {
            self.cpu_freqs.push(freq);
        }

        if let Some(power) = metrics.power_watts {
            self.power.push(power);
        }

        if let Some(util) = metrics.cpu_utilization {
            self.cpu_utilizations.push(util);
        }

        if let Some(util) = metrics.gpu_utilization {
            self.gpu_utilizations.push(util);
        }

        if let Some(mem) = metrics.memory_used_gb {
            self.memory_used.push(mem);
        }
    }

    pub fn get_cpu_temp_dataset(&self) -> Vec<(f64, f64)> {
        self.timestamps
            .iter()
            .zip(self.cpu_temps.iter())
            .map(|(&t, &v)| (t, v))
            .collect()
    }

    pub fn get_gpu_temp_dataset(&self) -> Vec<(f64, f64)> {
        self.timestamps
            .iter()
            .zip(self.gpu_temps.iter())
            .map(|(&t, &v)| (t, v))
            .collect()
    }

    pub fn get_cpu_temp_peak(&self) -> Option<f64> {
        self.cpu_temps
            .iter()
            .copied()
            .fold(None, |max, val| Some(max.map_or(val, |m| f64::max(m, val))))
    }

    pub fn get_cpu_temp_average(&self) -> Option<f64> {
        if self.cpu_temps.is_empty() {
            None
        } else {
            let sum: f64 = self.cpu_temps.iter().sum();
            Some(sum / self.cpu_temps.len() as f64)
        }
    }

    pub fn get_gpu_temp_peak(&self) -> Option<f64> {
        self.gpu_temps
            .iter()
            .copied()
            .fold(None, |max, val| Some(max.map_or(val, |m| f64::max(m, val))))
    }

    pub fn get_gpu_temp_average(&self) -> Option<f64> {
        if self.gpu_temps.is_empty() {
            None
        } else {
            let sum: f64 = self.gpu_temps.iter().sum();
            Some(sum / self.gpu_temps.len() as f64)
        }
    }

    pub fn get_x_bounds(&self) -> [f64; 2] {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let current_max = if elapsed < 20.0 {
            20.0
        } else if elapsed < self.max_duration_secs {
            elapsed.max(20.0)
        } else {
            self.max_duration_secs
        };
        [0.0, current_max]
    }

    fn calculate_nice_temp_scale(max_temp: f64) -> (f64, f64, Vec<f64>) {
        let y_min = 30.0;
        let y_max = ((max_temp.max(50.0) / 10.0).ceil() * 10.0).max(60.0);

        let range = y_max - y_min;
        let step = if range <= 50.0 {
            10.0
        } else if range <= 80.0 {
            20.0
        } else {
            25.0
        };

        let mut labels = Vec::new();
        let mut current = y_min;
        while current <= y_max {
            labels.push(current);
            current += step;
        }

        if labels.last() != Some(&y_max) {
            labels.push(y_max);
        }

        (y_min, y_max, labels)
    }
}

impl Default for MetricsHistory {
    fn default() -> Self {
        Self::new(300.0) // Default 5 minutes
    }
}

pub struct StressTestUI {
    history: MetricsHistory,
    cpu_active: bool,
    gpu_active: bool,
    cpu_primes: u64,
    gpu_iterations: u64,
    current_metrics: SystemMetrics,
}

impl StressTestUI {
    pub fn new(cpu_active: bool, gpu_active: bool, duration_secs: Option<u64>) -> Self {
        let duration = duration_secs.unwrap_or(300) as f64; // Default 5 minutes
        Self {
            history: MetricsHistory::new(duration),
            cpu_active,
            gpu_active,
            cpu_primes: 0,
            gpu_iterations: 0,
            current_metrics: SystemMetrics::default(),
        }
    }

    pub fn update_metrics(&mut self, metrics: SystemMetrics) {
        self.history.add_sample(&metrics);
        self.current_metrics = metrics;
    }

    pub fn update_workload_stats(&mut self, cpu_primes: u64, gpu_iterations: u64) {
        self.cpu_primes = cpu_primes;
        self.gpu_iterations = gpu_iterations;
    }

    pub fn render(&self, f: &mut Frame, metrics: &SystemMetrics) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Min(10),   // Graphs
                Constraint::Length(4), // System Metrics (NEW)
                Constraint::Length(4), // CPU Stats
                Constraint::Length(4), // GPU Stats
            ])
            .split(f.area());

        self.render_header(f, chunks[0]);
        self.render_graphs(f, chunks[1]);
        self.render_system_metrics(f, chunks[2], metrics);
        self.render_cpu_stats(f, chunks[3], metrics);
        self.render_gpu_stats(f, chunks[4], metrics);
    }

    fn render_header(&self, f: &mut Frame, area: Rect) {
        let now = Local::now().format("%Y-%m-%d %H:%M:%S");
        let status = if self.cpu_active && self.gpu_active {
            "CPU + GPU Stress Test"
        } else if self.cpu_active {
            "CPU Stress Test"
        } else {
            "GPU Stress Test"
        };

        let header = Paragraph::new(vec![Line::from(vec![Span::styled(
            format!("{} | {} | Press 'q' to quit", status, now),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )])])
        .block(Block::default().borders(Borders::ALL).title("iStress"));

        f.render_widget(header, area);
    }

    fn render_graphs(&self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        let x_bounds = self.history.get_x_bounds();

        let cpu_data = self.history.get_cpu_temp_dataset();
        if !cpu_data.is_empty() {
            let dataset = Dataset::default()
                .name("CPU Temp")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Red))
                .data(&cpu_data);

            let max_temp = cpu_data.iter().map(|(_, y)| *y).fold(0.0f64, f64::max);

            let (y_min, y_max, temp_labels) = MetricsHistory::calculate_nice_temp_scale(max_temp);

            let x_labels = vec![
                Span::raw("0"),
                Span::raw(format!("{:.0}", x_bounds[1] / 2.0)),
                Span::raw(format!("{:.0}", x_bounds[1])),
            ];

            let y_labels: Vec<Span> = temp_labels
                .iter()
                .map(|&t| Span::raw(format!("{}°", t as i32)))
                .collect();

            let chart = Chart::new(vec![dataset])
                .block(
                    Block::default()
                        .title("CPU Temperature (°C)")
                        .borders(Borders::ALL)
                        .style(Style::default().fg(Color::White)),
                )
                .x_axis(
                    Axis::default()
                        .title("Time (seconds)")
                        .style(Style::default().fg(Color::Gray))
                        .labels(x_labels)
                        .bounds(x_bounds),
                )
                .y_axis(
                    Axis::default()
                        .title("Temperature")
                        .style(Style::default().fg(Color::Gray))
                        .labels(y_labels)
                        .bounds([y_min, y_max]),
                );

            f.render_widget(chart, chunks[0]);
        }

        let gpu_data = self.history.get_gpu_temp_dataset();
        if !gpu_data.is_empty() {
            let dataset = Dataset::default()
                .name("GPU Temp")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Cyan))
                .data(&gpu_data);

            let max_temp = gpu_data.iter().map(|(_, y)| *y).fold(0.0f64, f64::max);

            let (y_min, y_max, temp_labels) = MetricsHistory::calculate_nice_temp_scale(max_temp);

            let x_labels = vec![
                Span::raw("0"),
                Span::raw(format!("{:.0}", x_bounds[1] / 2.0)),
                Span::raw(format!("{:.0}", x_bounds[1])),
            ];

            let y_labels: Vec<Span> = temp_labels
                .iter()
                .map(|&t| Span::raw(format!("{}°", t as i32)))
                .collect();

            let chart = Chart::new(vec![dataset])
                .block(
                    Block::default()
                        .title("GPU Temperature (°C)")
                        .borders(Borders::ALL)
                        .style(Style::default().fg(Color::White)),
                )
                .x_axis(
                    Axis::default()
                        .title("Time (seconds)")
                        .style(Style::default().fg(Color::Gray))
                        .labels(x_labels)
                        .bounds(x_bounds),
                )
                .y_axis(
                    Axis::default()
                        .title("Temperature")
                        .style(Style::default().fg(Color::Gray))
                        .labels(y_labels)
                        .bounds([y_min, y_max]),
                );

            f.render_widget(chart, chunks[1]);
        }
    }

    fn render_system_metrics(&self, f: &mut Frame, area: Rect, metrics: &SystemMetrics) {
        let cpu_util_str = metrics
            .cpu_utilization
            .map(|u| format!("{:.1}%", u))
            .unwrap_or_else(|| "N/A".to_string());

        let gpu_util_str = metrics
            .gpu_utilization
            .map(|u| format!("{:.1}%", u))
            .unwrap_or_else(|| "N/A".to_string());

        let memory_str =
            if let (Some(used), Some(total)) = (metrics.memory_used_gb, metrics.memory_total_gb) {
                format!("{:.1}/{:.1} GB", used, total)
            } else {
                "N/A".to_string()
            };

        let throttle_indicator = |throttled: Option<bool>| -> (String, Color) {
            match throttled {
                Some(true) => ("THROTTLED".to_string(), Color::Red),
                Some(false) => ("Normal".to_string(), Color::Green),
                None => ("Unknown".to_string(), Color::Gray),
            }
        };

        let (cpu_throttle_str, cpu_throttle_color) = throttle_indicator(metrics.cpu_throttled);
        let (gpu_throttle_str, gpu_throttle_color) = throttle_indicator(metrics.gpu_throttled);

        let info = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("CPU Util: ", Style::default().fg(Color::White)),
                Span::styled(cpu_util_str, Style::default().fg(Color::Cyan)),
                Span::raw("  |  "),
                Span::styled("GPU Util: ", Style::default().fg(Color::White)),
                Span::styled(gpu_util_str, Style::default().fg(Color::Cyan)),
            ]),
            Line::from(vec![
                Span::styled("Memory: ", Style::default().fg(Color::White)),
                Span::styled(memory_str, Style::default().fg(Color::Yellow)),
                Span::raw("  |  "),
                Span::styled("CPU: ", Style::default().fg(Color::White)),
                Span::styled(cpu_throttle_str, Style::default().fg(cpu_throttle_color)),
                Span::raw("  GPU: "),
                Span::styled(gpu_throttle_str, Style::default().fg(gpu_throttle_color)),
            ]),
        ])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("System Metrics"),
        );

        f.render_widget(info, area);
    }

    fn render_cpu_stats(&self, f: &mut Frame, area: Rect, metrics: &SystemMetrics) {
        let cpu_temp_str = metrics
            .cpu_temperature
            .map(|t| format!("{:.1}°C", t))
            .unwrap_or_else(|| "N/A".to_string());

        let peak_str = self
            .history
            .get_cpu_temp_peak()
            .map(|t| format!("{:.1}°C", t))
            .unwrap_or_else(|| "N/A".to_string());

        let avg_str = self
            .history
            .get_cpu_temp_average()
            .map(|t| format!("{:.1}°C", t))
            .unwrap_or_else(|| "N/A".to_string());

        let cpu_freq_str = metrics
            .cpu_frequency
            .map(|f| format!("{:.2} GHz", f))
            .unwrap_or_else(|| "N/A".to_string());

        let cpu_info = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Temp: ", Style::default().fg(Color::White)),
                Span::styled(cpu_temp_str, Style::default().fg(Color::Yellow)),
                Span::raw(" (Peak: "),
                Span::styled(peak_str, Style::default().fg(Color::Red)),
                Span::raw(", Avg: "),
                Span::styled(avg_str, Style::default().fg(Color::Cyan)),
                Span::raw(")"),
            ]),
            Line::from(vec![
                Span::styled("Frequency: ", Style::default().fg(Color::White)),
                Span::styled(cpu_freq_str, Style::default().fg(Color::Cyan)),
                Span::raw("  |  "),
                Span::styled("Primes: ", Style::default().fg(Color::White)),
                Span::styled(
                    format!("{}", self.cpu_primes),
                    Style::default().fg(Color::Green),
                ),
            ]),
        ])
        .block(Block::default().borders(Borders::ALL).title("CPU Stats"));

        f.render_widget(cpu_info, area);
    }

    fn render_gpu_stats(&self, f: &mut Frame, area: Rect, metrics: &SystemMetrics) {
        let gpu_temp_str = metrics
            .gpu_temperature
            .map(|t| format!("{:.1}°C", t))
            .unwrap_or_else(|| "N/A".to_string());

        let peak_str = self
            .history
            .get_gpu_temp_peak()
            .map(|t| format!("{:.1}°C", t))
            .unwrap_or_else(|| "N/A".to_string());

        let avg_str = self
            .history
            .get_gpu_temp_average()
            .map(|t| format!("{:.1}°C", t))
            .unwrap_or_else(|| "N/A".to_string());

        let power_str = metrics
            .power_watts
            .map(|p| format!("{:.1}W", p))
            .unwrap_or_else(|| "N/A".to_string());

        let gpu_freq_str = metrics
            .gpu_frequency
            .map(|f| format!("{:.2} GHz", f))
            .unwrap_or_else(|| "N/A".to_string());

        let gpu_info = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Temp: ", Style::default().fg(Color::White)),
                Span::styled(gpu_temp_str, Style::default().fg(Color::Yellow)),
                Span::raw(" (Peak: "),
                Span::styled(peak_str, Style::default().fg(Color::Red)),
                Span::raw(", Avg: "),
                Span::styled(avg_str, Style::default().fg(Color::Cyan)),
                Span::raw(")"),
            ]),
            Line::from(vec![
                Span::styled("Frequency: ", Style::default().fg(Color::White)),
                Span::styled(gpu_freq_str, Style::default().fg(Color::Cyan)),
                Span::raw("  |  "),
                Span::styled("Power: ", Style::default().fg(Color::White)),
                Span::styled(power_str, Style::default().fg(Color::Magenta)),
                Span::raw("  |  "),
                Span::styled("Iterations: ", Style::default().fg(Color::White)),
                Span::styled(
                    format!("{}", self.gpu_iterations),
                    Style::default().fg(Color::Green),
                ),
            ]),
        ])
        .block(Block::default().borders(Borders::ALL).title("GPU Stats"));

        f.render_widget(gpu_info, area);
    }

    pub fn generate_report(&self, actual_duration_secs: f64) -> String {
        let mut report = String::new();

        report.push('\n');
        report.push_str("═══════════════════════════════════════════════════════════\n");
        report.push_str("                 iStress - Final Test Report                \n");
        report.push_str("═══════════════════════════════════════════════════════════\n\n");

        report.push_str("Test Configuration:\n");
        report.push_str("───────────────────────────────────────────────────────────\n");
        report.push_str(&format!(
            "  Duration: {:.1} seconds ({:.1} minutes)\n",
            actual_duration_secs,
            actual_duration_secs / 60.0
        ));
        report.push_str(&format!(
            "  CPU Test: {}\n",
            if self.cpu_active {
                "Enabled"
            } else {
                "Disabled"
            }
        ));
        report.push_str(&format!(
            "  GPU Test: {}\n\n",
            if self.gpu_active {
                "Enabled"
            } else {
                "Disabled"
            }
        ));

        if self.cpu_active {
            report.push_str("CPU Statistics:\n");
            report.push_str("───────────────────────────────────────────────────────────\n");

            if let Some(peak) = self.history.get_cpu_temp_peak() {
                report.push_str(&format!("  Peak Temperature: {:.1}°C\n", peak));
            }
            if let Some(avg) = self.history.get_cpu_temp_average() {
                report.push_str(&format!("  Average Temperature: {:.1}°C\n", avg));
            }
            if let Some(current) = self.current_metrics.cpu_temperature {
                report.push_str(&format!("  Final Temperature: {:.1}°C\n", current));
            }
            if let Some(freq) = self.current_metrics.cpu_frequency {
                report.push_str(&format!("  CPU Frequency: {:.2} GHz\n", freq));
            }
            if let Some(util) = self.current_metrics.cpu_utilization {
                report.push_str(&format!("  CPU Utilization: {:.1}%\n", util));
            }
            report.push_str(&format!("  Primes Found: {}\n", self.cpu_primes));
            if actual_duration_secs > 0.0 {
                report.push_str(&format!(
                    "  Primes/Second: {:.0}\n",
                    self.cpu_primes as f64 / actual_duration_secs
                ));
            }

            match self.current_metrics.cpu_throttled {
                Some(true) => report.push_str("  Throttling: DETECTED\n"),
                Some(false) => report.push_str("  Throttling: None\n"),
                None => report.push_str("  Throttling: Unknown\n"),
            }
            report.push('\n');
        }

        if self.gpu_active {
            report.push_str("GPU Statistics:\n");
            report.push_str("───────────────────────────────────────────────────────────\n");

            if let Some(peak) = self.history.get_gpu_temp_peak() {
                report.push_str(&format!("  Peak Temperature: {:.1}°C\n", peak));
            }
            if let Some(avg) = self.history.get_gpu_temp_average() {
                report.push_str(&format!("  Average Temperature: {:.1}°C\n", avg));
            }
            if let Some(current) = self.current_metrics.gpu_temperature {
                report.push_str(&format!("  Final Temperature: {:.1}°C\n", current));
            }
            if let Some(freq) = self.current_metrics.gpu_frequency {
                report.push_str(&format!("  GPU Frequency: {:.2} GHz\n", freq));
            }
            if let Some(util) = self.current_metrics.gpu_utilization {
                report.push_str(&format!("  GPU Utilization: {:.1}%\n", util));
            }
            if let Some(power) = self.current_metrics.power_watts {
                report.push_str(&format!("  Power Consumption: {:.1}W\n", power));
            }
            report.push_str(&format!("  GPU Iterations: {}\n", self.gpu_iterations));
            if actual_duration_secs > 0.0 {
                report.push_str(&format!(
                    "  Iterations/Second: {:.0}\n",
                    self.gpu_iterations as f64 / actual_duration_secs
                ));
            }

            match self.current_metrics.gpu_throttled {
                Some(true) => report.push_str("  Throttling: DETECTED\n"),
                Some(false) => report.push_str("  Throttling: None\n"),
                None => report.push_str("  Throttling: Unknown\n"),
            }
            report.push('\n');
        }

        report.push_str("System Statistics:\n");
        report.push_str("───────────────────────────────────────────────────────────\n");
        if let (Some(used), Some(total)) = (
            self.current_metrics.memory_used_gb,
            self.current_metrics.memory_total_gb,
        ) {
            report.push_str(&format!(
                "  Memory Used: {:.1}/{:.1} GB ({:.1}%)\n",
                used,
                total,
                (used / total) * 100.0
            ));
        }

        report.push('\n');
        report.push_str("═══════════════════════════════════════════════════════════\n");
        report.push_str("                       Test Complete                        \n");
        report.push_str("═══════════════════════════════════════════════════════════\n");

        report
    }
}

pub fn run_ui<F>(mut ui: StressTestUI, mut update_fn: F) -> io::Result<StressTestUI>
where
    F: FnMut(&mut StressTestUI) -> bool,
{
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_ui_loop(&mut terminal, &mut ui, &mut update_fn);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result?;
    Ok(ui)
}

fn run_ui_loop<B: Backend, F>(
    terminal: &mut Terminal<B>,
    ui: &mut StressTestUI,
    update_fn: &mut F,
) -> io::Result<()>
where
    F: FnMut(&mut StressTestUI) -> bool,
{
    let mut last_update = Instant::now();
    let update_interval = Duration::from_millis(500);

    loop {
        if event::poll(Duration::from_millis(10))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press && key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }

        if last_update.elapsed() >= update_interval {
            let should_continue = update_fn(ui);
            if !should_continue {
                break;
            }

            let metrics = ui.current_metrics.clone();
            terminal.draw(|f| ui.render(f, &metrics))?;

            last_update = Instant::now();
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_history_creation() {
        let history = MetricsHistory::new(60.0);
        assert_eq!(history.timestamps.len(), 0);
        assert_eq!(history.cpu_temps.len(), 0);
        assert_eq!(history.max_duration_secs, 60.0);
    }

    #[test]
    fn test_metrics_history_add_sample() {
        let mut history = MetricsHistory::new(60.0);
        let metrics = SystemMetrics {
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

        history.add_sample(&metrics);

        assert_eq!(history.timestamps.len(), 1);
        assert_eq!(history.cpu_temps.len(), 1);
        assert_eq!(history.cpu_temps[0], 65.5);
    }

    #[test]
    fn test_metrics_history_no_limit() {
        let mut history = MetricsHistory::new(300.0);
        let metrics = SystemMetrics {
            cpu_temperature: Some(60.0),
            gpu_temperature: None,
            cpu_frequency: None,
            gpu_frequency: None,
            power_watts: None,
            cpu_utilization: None,
            gpu_utilization: None,
            memory_used_gb: None,
            memory_total_gb: None,
            cpu_throttled: None,
            gpu_throttled: None,
        };

        for _ in 0..200 {
            history.add_sample(&metrics);
        }

        assert_eq!(history.timestamps.len(), 200);
        assert_eq!(history.cpu_temps.len(), 200);
    }

    #[test]
    fn test_stress_test_ui_creation() {
        let ui = StressTestUI::new(true, false, Some(60));
        assert!(ui.cpu_active);
        assert!(!ui.gpu_active);
        assert_eq!(ui.cpu_primes, 0);
        assert_eq!(ui.gpu_iterations, 0);
    }

    #[test]
    fn test_stress_test_ui_update_workload_stats() {
        let mut ui = StressTestUI::new(true, true, Some(60));
        ui.update_workload_stats(1000, 500);
        assert_eq!(ui.cpu_primes, 1000);
        assert_eq!(ui.gpu_iterations, 500);
    }

    #[test]
    fn test_stress_test_ui_update_metrics() {
        let mut ui = StressTestUI::new(true, false, Some(60));
        let metrics = SystemMetrics {
            cpu_temperature: Some(70.0),
            gpu_temperature: Some(60.0),
            cpu_frequency: Some(3.5),
            gpu_frequency: Some(1.5),
            power_watts: Some(30.0),
            cpu_utilization: None,
            gpu_utilization: None,
            memory_used_gb: None,
            memory_total_gb: None,
            cpu_throttled: None,
            gpu_throttled: None,
        };

        ui.update_metrics(metrics);
        assert_eq!(ui.history.cpu_temps.len(), 1);
        assert_eq!(ui.history.cpu_temps[0], 70.0);
    }

    #[test]
    fn test_get_datasets() {
        let mut history = MetricsHistory::new(300.0);

        for i in 0..5 {
            let metrics = SystemMetrics {
                cpu_temperature: Some(60.0 + i as f64),
                gpu_temperature: Some(50.0 + i as f64),
                cpu_frequency: None,
                gpu_frequency: None,
                power_watts: None,
                cpu_utilization: None,
                gpu_utilization: None,
                memory_used_gb: None,
                memory_total_gb: None,
                cpu_throttled: None,
                gpu_throttled: None,
            };
            history.add_sample(&metrics);
            std::thread::sleep(Duration::from_millis(10));
        }

        let cpu_dataset = history.get_cpu_temp_dataset();
        let gpu_dataset = history.get_gpu_temp_dataset();

        assert_eq!(cpu_dataset.len(), 5);
        assert_eq!(gpu_dataset.len(), 5);
    }
}
