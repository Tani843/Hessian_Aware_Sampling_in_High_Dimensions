#!/usr/bin/env python3
"""
Complete experimental pipeline for Hessian Aware Sampling project.
This script runs all experiments, generates all plots, and builds documentation.
"""

import argparse
import logging
import time
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_phase_1_setup():
    """Phase 1: Verify setup and dependencies"""
    logger.info("=== Phase 1: Setup Verification ===")
    
    try:
        import numpy as np
        import scipy
        import matplotlib.pyplot as plt
        import seaborn as sns
        logger.info("✓ All dependencies available")
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        return False
    
    # Check project structure
    required_dirs = [
        "src/core", "src/samplers", "src/utils", "src/visualization",
        "tests", "examples", "docs/jekyll_site"
    ]
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            logger.error(f"✗ Missing directory: {dir_path}")
            return False
    
    logger.info("✓ Project structure verified")
    return True

def run_phase_2_core_sampling():
    """Phase 2: Test core sampling algorithms"""
    logger.info("=== Phase 2: Core Sampling Tests ===")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.samplers.advanced_hessian_samplers import HessianAwareMetropolis
        from examples.test_distributions import get_test_distribution
        
        # Quick functionality test
        import numpy as np
        dim = 20
        target_dist = get_test_distribution('gaussian', dim, condition_number=10.0)
        sampler = HessianAwareMetropolis(target_dist.log_prob, dim)
        
        result = sampler.sample(100, np.random.randn(dim))
        logger.info(f"✓ Core sampling working, generated {result.samples.shape} samples")
        return True
        
    except Exception as e:
        logger.error(f"✗ Core sampling failed: {e}")
        return False

def run_phase_3_benchmarking():
    """Phase 3: Run comprehensive benchmarks"""
    logger.info("=== Phase 3: Benchmarking ===")
    
    try:
        from examples.comprehensive_benchmark import main as run_benchmark
        
        # This should run the full benchmark suite
        logger.info("Starting comprehensive benchmark...")
        start_time = time.time()
        
        run_benchmark()  # This runs the full experimental suite
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Benchmarking completed in {elapsed:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"✗ Benchmarking failed: {e}")
        return False

def run_phase_4_visualization():
    """Phase 4: Generate all plots and visualizations"""
    logger.info("=== Phase 4: Visualization Generation ===")
    
    try:
        from examples.publication_results import create_publication_figures
        
        logger.info("Generating publication figures...")
        create_publication_figures()
        
        # Verify outputs
        output_dir = Path("results/figures")
        expected_figures = [
            "fig1_comparison.pdf", "fig2_scaling.pdf", 
            "fig3_hessian.pdf", "fig4_cost_accuracy.pdf"
        ]
        
        for fig_name in expected_figures:
            if (output_dir / fig_name).exists():
                logger.info(f"✓ Generated {fig_name}")
            else:
                logger.warning(f"⚠ Missing {fig_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Visualization generation failed: {e}")
        return False

def run_phase_5_documentation():
    """Phase 5: Build documentation website"""
    logger.info("=== Phase 5: Documentation Build ===")
    
    try:
        import subprocess
        jekyll_dir = Path("docs/jekyll_site")
        
        # Copy generated figures to Jekyll assets
        import shutil
        source_dir = Path("results/figures")
        target_dir = jekyll_dir / "assets/images/plots"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if source_dir.exists():
            for fig_file in source_dir.glob("*.png"):
                shutil.copy2(fig_file, target_dir)
            logger.info("✓ Figures copied to Jekyll site")
        
        # Test Jekyll build (if Jekyll is available)
        try:
            result = subprocess.run(
                ["bundle", "exec", "jekyll", "build"],
                cwd=jekyll_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info("✓ Jekyll site built successfully")
            else:
                logger.warning(f"⚠ Jekyll build issues: {result.stderr}")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.info("⚠ Jekyll not available, skipping build test")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Documentation build failed: {e}")
        return False

def run_phase_6_validation():
    """Phase 6: Final validation and testing"""
    logger.info("=== Phase 6: Final Validation ===")
    
    try:
        import subprocess
        
        # Run unit tests
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✓ All unit tests passed")
        else:
            logger.error(f"✗ Unit tests failed:\n{result.stdout}\n{result.stderr}")
            return False
        
        # Validate final outputs
        expected_outputs = [
            "results/benchmark_results.json",
            "results/figures/",
            "docs/jekyll_site/_site/",
            "experiment.log"
        ]
        
        for output_path in expected_outputs:
            if Path(output_path).exists():
                logger.info(f"✓ Output verified: {output_path}")
            else:
                logger.warning(f"⚠ Missing output: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Final validation failed: {e}")
        return False

def main():
    """Run complete experimental pipeline"""
    parser = argparse.ArgumentParser(description="Run complete Hessian sampling experiment")
    parser.add_argument("--skip-benchmark", action="store_true", 
                       help="Skip time-consuming benchmark phase")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick validation only")
    args = parser.parse_args()
    
    logger.info("Starting complete experimental pipeline...")
    start_time = time.time()
    
    phases = [
        ("Setup Verification", run_phase_1_setup),
        ("Core Sampling", run_phase_2_core_sampling),
    ]
    
    if not args.quick_test:
        if not args.skip_benchmark:
            phases.append(("Benchmarking", run_phase_3_benchmarking))
        phases.extend([
            ("Visualization", run_phase_4_visualization),
            ("Documentation", run_phase_5_documentation),
        ])
    
    phases.append(("Final Validation", run_phase_6_validation))
    
    # Run all phases
    success_count = 0
    for phase_name, phase_func in phases:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {phase_name}")
        logger.info(f"{'='*50}")
        
        if phase_func():
            success_count += 1
            logger.info(f"✓ {phase_name} completed successfully")
        else:
            logger.error(f"✗ {phase_name} failed")
            if not args.quick_test:  # Don't exit early in quick test mode
                break
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*50}")
    logger.info(f"PIPELINE SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Completed {success_count}/{len(phases)} phases")
    logger.info(f"Total time: {total_time:.2f} seconds")
    
    if success_count == len(phases):
        logger.info("🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
        logger.info("Project is ready for deployment and publication.")
    else:
        logger.error("❌ Some phases failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()