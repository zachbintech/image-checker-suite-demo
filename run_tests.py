#!/usr/bin/env python3
"""
Test runner for the image processing suite.
This script runs all tests individually to avoid import issues with unittest discovery.
"""

import sys
import unittest
import os

# Add current directory to Python path
sys.path.insert(0, '.')

def run_all_tests():
    """Run all tests in the test suite."""
    print("🧪 Running Image Processing Suite Tests")
    print("=" * 50)
    
    # List of all test modules
    test_modules = [
        'tests.test_artifacts',
        'tests.test_image_quality', 
        'tests.test_image_utilities',
        'tests.test_app_integration',
        'tests.test_dino_embeddings',
        'tests.test_groupings',
        'tests.test_similarity'
    ]
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    loaded_modules = []
    failed_modules = []
    
    # Load each test module
    for module_name in test_modules:
        try:
            tests = loader.loadTestsFromName(module_name)
            suite.addTests(tests)
            loaded_modules.append(module_name)
            print(f"✅ Loaded {module_name}")
        except Exception as e:
            failed_modules.append((module_name, str(e)))
            print(f"❌ Failed to load {module_name}: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Loading Summary:")
    print(f"   Loaded: {len(loaded_modules)}/{len(test_modules)} modules")
    
    if failed_modules:
        print(f"   Failed: {len(failed_modules)} modules")
        for module, error in failed_modules:
            print(f"     - {module}: {error}")
    
    print("\n🏃 Running Tests...")
    print("=" * 50)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"📈 Test Results Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    # Calculate skipped count safely
    skipped_count = len(result.skipped) if hasattr(result, 'skipped') and result.skipped else 0
    print(f"   Skipped: {skipped_count}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   - {test}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   - {test}")
    
    # Return success/failure
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print(f"\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  Some tests failed or had errors.")
    
    return success

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)