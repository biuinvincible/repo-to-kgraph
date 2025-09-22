#!/usr/bin/env python3
"""
Parser Verification Test Runner

This script runs comprehensive verification tests on our parser system
using manually crafted test code where every entity and relationship
is known and verifiable.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tests.test_parser_verification import test_parser_comprehensive_verification


async def main():
    """Run the verification test."""
    print("🧪 Starting Parser Verification Test Suite")
    print("=" * 50)
    print("This test verifies our parser can extract entities and relationships")
    print("from manually crafted code with 100% known expected results.")
    print("=" * 50)

    try:
        await test_parser_comprehensive_verification()
        print("\n✅ Verification test completed successfully!")
        return 0

    except Exception as e:
        print(f"\n❌ Verification test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)