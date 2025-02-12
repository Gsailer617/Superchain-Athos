#!/usr/bin/env python3

import pytest
import asyncio
import json
import os
import sys
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from dataclasses import dataclass, asdict
import traceback
import signal
import time
import coverage
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_run.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test execution result"""
    suite: str
    start_time: str
    end_time: str
    duration: str
    exit_code: int
    success: bool
    error_message: Optional[str] = None
    coverage_percentage: Optional[float] = None
    skipped_tests: Optional[int] = None
    failed_tests: Optional[int] = None
    total_tests: Optional[int] = None
    passed_tests: Optional[int] = None
    warnings: Optional[List[str]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    resource_usage: Optional[Dict[str, float]] = None
    test_categories: Optional[Dict[str, int]] = None
    flaky_tests: Optional[List[str]] = None
    slow_tests: Optional[List[Dict[str, Any]]] = None

@dataclass
class TestMetrics:
    """Test execution metrics"""
    total_duration: float
    avg_test_duration: float
    max_test_duration: float
    min_test_duration: float
    memory_usage: float
    cpu_usage: float
    test_count: int
    assertion_count: int
    error_count: int
    warning_count: int

@dataclass
class TestSuite:
    """Test suite configuration"""
    name: str
    path: str
    markers: List[str]
    required_services: List[str]
    timeout: int = 300  # 5 minutes default timeout

class TestRunner:
    """Enhanced test runner for executing test suites"""
    
    def __init__(self):
        self.test_config = self._load_config()
        self.results: Dict[str, TestResult] = {}
        self.coverage = coverage.Coverage()
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        
    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Handle interrupt signals"""
        logger.warning("Received interrupt signal. Cleaning up...")
        self._cleanup_test_environment()
        sys.exit(1)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load test configuration with error handling"""
        try:
            config_path = Path(__file__).parent / 'config' / 'test_config.json'
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
                
            with open(config_path) as f:
                config = json.load(f)
                
            self._validate_config(config)
            return config
            
        except Exception as e:
            logger.error(f"Error loading test configuration: {e}")
            raise
            
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate test configuration"""
        required_keys = {'test_environment', 'circuit_breaker', 'performance'}
        missing_keys = required_keys - set(config.keys())
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
            
    def _setup_test_environment(self) -> None:
        """Setup test environment with error handling"""
        try:
            # Create necessary directories
            os.makedirs('tests/data/monitoring', exist_ok=True)
            os.makedirs('tests/reports', exist_ok=True)
            
            # Set environment variables
            os.environ['TEST_MODE'] = 'true'
            os.environ['PROMETHEUS_PORT'] = str(
                self.test_config['test_environment']['monitoring']['prometheus_port']
            )
            
            # Start coverage collection
            self.coverage.start()
            
        except Exception as e:
            logger.error(f"Error setting up test environment: {e}")
            raise
            
    def _cleanup_test_environment(self) -> None:
        """Cleanup test environment with error handling"""
        try:
            # Stop coverage collection
            self.coverage.stop()
            self.coverage.save()
            
            # Remove test data
            import shutil
            shutil.rmtree('tests/data/monitoring', ignore_errors=True)
            
        except Exception as e:
            logger.error(f"Error cleaning up test environment: {e}")
            
    def _get_test_suites(self) -> List[TestSuite]:
        """Get configured test suites"""
        return [
            TestSuite(
                name='unit',
                path='tests/unit',
                markers=['unit'],
                required_services=[],
                timeout=60
            ),
            TestSuite(
                name='integration',
                path='tests/integration',
                markers=['integration'],
                required_services=['redis', 'prometheus'],
                timeout=300
            ),
            TestSuite(
                name='performance',
                path='tests/performance',
                markers=['benchmark'],
                required_services=['redis'],
                timeout=600
            ),
            TestSuite(
                name='security',
                path='tests/security',
                markers=['security'],
                required_services=[],
                timeout=300
            ),
            TestSuite(
                name='chaos',
                path='tests/chaos',
                markers=['chaos'],
                required_services=['redis', 'prometheus'],
                timeout=900
            ),
            TestSuite(
                name='contract',
                path='tests/contracts',
                markers=['contract'],
                required_services=[],
                timeout=300
            )
        ]
        
    def _check_required_services(self, suite: TestSuite) -> bool:
        """Check if required services are available"""
        for service in suite.required_services:
            if service == 'redis':
                try:
                    import redis
                    client = redis.Redis(host='localhost', port=6379, db=0)
                    client.ping()
                except Exception:
                    logger.error(f"Redis not available for {suite.name}")
                    return False
            elif service == 'prometheus':
                try:
                    import requests
                    response = requests.get('http://localhost:9090/-/healthy')
                    if response.status_code != 200:
                        raise Exception("Prometheus not healthy")
                except Exception:
                    logger.error(f"Prometheus not available for {suite.name}")
                    return False
        return True
        
    async def run_tests(self, suite_name: str = 'all', verbose: bool = False) -> bool:
        """Run specified test suite with enhanced error handling"""
        try:
            self._setup_test_environment()
            
            suites = self._get_test_suites()
            if suite_name != 'all':
                suites = [s for s in suites if s.name == suite_name]
                if not suites:
                    raise ValueError(f"Invalid test suite: {suite_name}")
                    
            success = True
            for suite in suites:
                if not self._check_required_services(suite):
                    logger.error(f"Skipping {suite.name} due to missing services")
                    continue
                    
                # Prepare pytest arguments
                pytest_args = ['-v'] if verbose else []
                pytest_args.extend([
                    suite.path,
                    *[f"-m {marker}" for marker in suite.markers],
                    '--asyncio-mode=auto',
                    '--capture=no' if verbose else '--capture=fd',
                    '--log-cli-level=INFO' if verbose else '--log-cli-level=WARNING',
                    f'--timeout={suite.timeout}'
                ])
                
                # Run tests with timeout
                start_time = datetime.now()
                try:
                    result = pytest.main(pytest_args)
                    success = success and result == pytest.ExitCode.OK
                except Exception as e:
                    logger.error(f"Error running {suite.name} tests: {e}")
                    success = False
                    result = 1
                    
                end_time = datetime.now()
                
                # Calculate coverage
                self.coverage.stop()
                coverage_data = self.coverage.report(include=f"src/{suite.name}/*")
                self.coverage.start()
                
                # Store results
                self.results[suite.name] = TestResult(
                    suite=suite.name,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration=str(end_time - start_time),
                    exit_code=result,
                    success=result == pytest.ExitCode.OK,
                    coverage_percentage=coverage_data,
                    error_message=traceback.format_exc() if result != pytest.ExitCode.OK else None
                )
                
            return success
            
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False
            
        finally:
            self._cleanup_test_environment()
            
    def generate_report(self, output_file: Optional[str] = None) -> None:
        """Generate detailed test report"""
        if not self.results:
            logger.warning("No test results available")
            return
            
        # Calculate overall metrics
        total_duration = sum(
            (datetime.fromisoformat(r.end_time) - datetime.fromisoformat(r.start_time)).total_seconds()
            for r in self.results.values()
        )
        total_tests = sum(r.total_tests or 0 for r in self.results.values())
        total_passed = sum(r.passed_tests or 0 for r in self.results.values())
        total_failed = sum(r.failed_tests or 0 for r in self.results.values())
        total_skipped = sum(r.skipped_tests or 0 for r in self.results.values())
        
        # Generate test categories summary
        category_summary = {}
        for result in self.results.values():
            if result.test_categories:
                for category, count in result.test_categories.items():
                    category_summary[category] = category_summary.get(category, 0) + count
                    
        # Generate performance summary
        performance_summary = {
            'slow_tests': [],
            'resource_intensive_tests': [],
            'flaky_tests': []
        }
        for result in self.results.values():
            if result.slow_tests:
                performance_summary['slow_tests'].extend(result.slow_tests)
            if result.flaky_tests:
                performance_summary['flaky_tests'].extend(result.flaky_tests)
            
        # Sort performance issues
        performance_summary['slow_tests'].sort(key=lambda x: x.get('duration', 0), reverse=True)
        
        report = {
            'summary': {
                'timestamp': datetime.now().isoformat(),
                'total_duration': total_duration,
                'total_suites': len(self.results),
                'successful_suites': sum(1 for r in self.results.values() if r.success),
                'total_tests': total_tests,
                'passed_tests': total_passed,
                'failed_tests': total_failed,
                'skipped_tests': total_skipped,
                'success_rate': (total_passed / total_tests * 100) if total_tests > 0 else 0,
                'total_coverage': self.coverage.report(show_missing=False),
                'test_categories': category_summary
            },
            'performance': {
                'slow_tests': performance_summary['slow_tests'][:10],  # Top 10 slowest tests
                'flaky_tests': performance_summary['flaky_tests'],
                'resource_usage': {
                    suite: result.resource_usage
                    for suite, result in self.results.items()
                    if result.resource_usage
                }
            },
            'test_results': {
                suite: {
                    'result': asdict(result),
                    'details': {
                        'categories': result.test_categories,
                        'performance': result.performance_metrics,
                        'warnings': result.warnings
                    }
                }
                for suite, result in self.results.items()
            },
            'environment': {
                'python_version': sys.version,
                'platform': sys.platform,
                'cpu_count': os.cpu_count(),
                'memory_available': psutil.virtual_memory().available,
                'test_mode': os.getenv('TEST_MODE'),
                'ci_environment': os.getenv('CI') == 'true'
            },
            'configuration': self.test_config
        }
        
        # Generate HTML report
        html_report = self._generate_html_report(report)
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save JSON report
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"JSON test report written to {output_file}")
            
            # Save HTML report
            html_path = output_path.with_suffix('.html')
            with open(html_path, 'w') as f:
                f.write(html_report)
            logger.info(f"HTML test report written to {html_path}")
        else:
            print(json.dumps(report, indent=2))

    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report from test results"""
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Execution Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .summary { background: #f5f5f5; padding: 20px; border-radius: 5px; }
                .success { color: green; }
                .failure { color: red; }
                .warning { color: orange; }
                .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
                .metric-card { background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .slow-tests { margin-top: 20px; }
                .test-suite { margin: 20px 0; }
                .chart { margin: 20px 0; }
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Test Execution Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>Test Results</h3>
                        <p>Total Tests: {total_tests}</p>
                        <p class="success">Passed: {passed_tests}</p>
                        <p class="failure">Failed: {failed_tests}</p>
                        <p class="warning">Skipped: {skipped_tests}</p>
                    </div>
                    <div class="metric-card">
                        <h3>Coverage</h3>
                        <p>Total Coverage: {coverage:.1f}%</p>
                    </div>
                    <div class="metric-card">
                        <h3>Performance</h3>
                        <p>Total Duration: {duration:.1f}s</p>
                        <p>Success Rate: {success_rate:.1f}%</p>
                    </div>
                </div>
            </div>
            
            <div class="test-suites">
                <h2>Test Suites</h2>
                {test_suites}
            </div>
            
            <div class="slow-tests">
                <h2>Slow Tests</h2>
                {slow_tests}
            </div>
            
            <div class="charts">
                <div id="coverageChart" class="chart"></div>
                <div id="performanceChart" class="chart"></div>
            </div>
            
            <script>
                // Add interactive charts using Plotly
                {charts_js}
            </script>
        </body>
        </html>
        """
        
        # Generate test suites HTML
        test_suites_html = ""
        for suite, data in report['test_results'].items():
            result = data['result']
            test_suites_html += f"""
            <div class="test-suite">
                <h3>{suite}</h3>
                <p>Status: <span class="{'success' if result['success'] else 'failure'}">{
                    'Passed' if result['success'] else 'Failed'}</span></p>
                <p>Duration: {result['duration']}</p>
                <p>Coverage: {result.get('coverage_percentage', 0):.1f}%</p>
            </div>
            """
        
        # Generate slow tests HTML
        slow_tests_html = "<ul>"
        for test in report['performance']['slow_tests']:
            slow_tests_html += f"<li>{test['name']} - {test['duration']:.2f}s</li>"
        slow_tests_html += "</ul>"
        
        # Generate charts JavaScript
        charts_js = """
        Plotly.newPlot('coverageChart', [{
            values: [%s],
            labels: ['Covered', 'Uncovered'],
            type: 'pie',
            title: 'Code Coverage'
        }]);
        
        Plotly.newPlot('performanceChart', [{
            x: [%s],
            y: [%s],
            type: 'bar',
            name: 'Test Duration'
        }], {
            title: 'Test Suite Performance'
        });
        """ % (
            report['summary']['total_coverage'],
            ','.join(f"'{s}'" for s in report['test_results'].keys()),
            ','.join(str(d['result']['duration']) for d in report['test_results'].values())
        )
        
        # Format the template
        return template.format(
            total_tests=report['summary']['total_tests'],
            passed_tests=report['summary']['passed_tests'],
            failed_tests=report['summary']['failed_tests'],
            skipped_tests=report['summary']['skipped_tests'],
            coverage=report['summary']['total_coverage'],
            duration=report['summary']['total_duration'],
            success_rate=report['summary']['success_rate'],
            test_suites=test_suites_html,
            slow_tests=slow_tests_html,
            charts_js=charts_js
        )

def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run test suites')
    parser.add_argument(
        '--suite',
        choices=['all', 'unit', 'integration', 'performance', 'security', 'chaos', 'contract'],
        default='all',
        help='Test suite to run'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--report',
        help='Output file for test report'
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    success = asyncio.run(runner.run_tests(args.suite, args.verbose))
    runner.generate_report(args.report)
    
    sys.exit(0 if success else 1)
    
if __name__ == '__main__':
    main() 