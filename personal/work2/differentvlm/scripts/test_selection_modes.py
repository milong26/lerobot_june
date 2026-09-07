#!/usr/bin/env python
"""
Test script to verify TinyVLA selection mode support.

Tests:
1. Config creation with different selection modes
2. Command-line argument parsing
3. Unified selection module import
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
WORK2_ROOT = Path(__file__).resolve().parents[2]
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))


def test_config_with_selection_modes():
    """Test creating configs with different selection modes."""
    from differentvlm.configs.vlm_config import get_config
    
    print("\n" + "="*60)
    print("Test 1: Config Creation with Different Selection Modes")
    print("="*60)
    
    for selection_mode in ["v5", "grid_uniform", "random"]:
        print(f"\nTesting selection_mode='{selection_mode}'...")
        
        cfg_s = get_config(
            vlm_name="tinyvla_s",
            gpu_id=0,
            dataset_name="pick_place-v3_corner",
            selection_mode=selection_mode
        )
        
        assert cfg_s.selection_mode == selection_mode, \
            f"Expected selection_mode='{selection_mode}', got '{cfg_s.selection_mode}'"
        assert cfg_s.selection_num_episodes == 112
        assert cfg_s.tinyvla_policy_type == "tinyvla_s"
        
        cfg_b = get_config(
            vlm_name="tinyvla_b",
            gpu_id=1,
            dataset_name="disassemble-v3_corner",
            selection_mode=selection_mode
        )
        
        assert cfg_b.selection_mode == selection_mode
        assert cfg_b.tinyvla_policy_type == "tinyvla_b"
        
        print(f"  ✓ tinyvla_s config created successfully")
        print(f"  ✓ tinyvla_b config created successfully")
        print(f"  ✓ selection_mode={cfg_s.selection_mode}")
        print(f"  ✓ experiment_dir={cfg_s.experiment_dir}")
    
    print("\n✓ All config tests passed!")
    return True


def test_unified_selection_import():
    """Test importing the unified selection module."""
    print("\n" + "="*60)
    print("Test 2: Unified Selection Module Import")
    print("="*60)
    
    try:
        from differentvlm.selection.unified_selection import run_selection
        print("  ✓ Successfully imported run_selection")
        
        import inspect
        sig = inspect.signature(run_selection)
        print(f"  ✓ Function signature: {sig}")
        
        print("\n✓ Unified selection module import test passed!")
        return True
    except Exception as e:
        print(f"  ✗ Failed to import: {e}")
        return False


def test_argument_parsing():
    """Test that the main script accepts the new arguments."""
    print("\n" + "="*60)
    print("Test 3: Argument Parsing")
    print("="*60)
    
    import subprocess
    
    test_script = Path(__file__).resolve().parent / "run_tinyvla.py"
    
    for mode in ["v5", "grid_uniform", "random"]:
        cmd = [
            sys.executable, str(test_script),
            "--help"
        ]
        
        print(f"\nTesting help output (should include --selection-mode)...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if "--selection-mode" in result.stdout:
            print(f"  ✓ --selection-mode argument found in help")
        else:
            print(f"  ✗ --selection-mode argument NOT found in help")
            return False
    
    print("\n✓ Argument parsing test passed!")
    return True


def main():
    print("\n" + "#"*60)
    print("# TinyVLA Selection Mode Tests")
    print("#"*60)
    
    tests = [
        ("Config Creation", test_config_with_selection_modes),
        ("Unified Selection Import", test_unified_selection_import),
        ("Argument Parsing", test_argument_parsing),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "#"*60)
        print("# All Tests Passed!")
        print("#"*60)
        return 0
    else:
        print("\n" + "#"*60)
        print("# Some Tests Failed!")
        print("#"*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())