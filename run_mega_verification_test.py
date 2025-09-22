#!/usr/bin/env python3
"""
MEGA Parser Verification Test Runner

This script runs the most comprehensive verification tests on our parser system
using manually crafted multi-language test code with advanced programming patterns.

Features tested:
- Multi-language parsing (Python, JavaScript, TypeScript)
- Complex inheritance and mixin patterns
- Generic programming constructs
- Decorator and annotation patterns
- Async/await patterns
- Cross-file dependencies
- Advanced relationship detection

Expected Results: 149+ entities, 250+ relationships
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tests.test_mega_verification import test_mega_parser_verification


def setup_logging():
    """Configure logging for the test run."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('mega_verification.log')
        ]
    )


async def main():
    """Run the MEGA verification test suite."""
    setup_logging()

    print("🚀 MEGA PARSER VERIFICATION TEST SUITE")
    print("=" * 60)
    print("Testing parser against complex multi-language codebase with:")
    print("  • Python (inheritance, decorators, async patterns)")
    print("  • JavaScript (ES6+, classes, promises)")
    print("  • TypeScript (generics, interfaces, advanced types)")
    print("  • Expected: 149+ entities, 250+ relationships")
    print("=" * 60)

    try:
        success = await test_mega_parser_verification()

        if success:
            print("\n" + "=" * 60)
            print("🎉 MEGA VERIFICATION TEST SUITE: ✅ PASSED")
            print("=" * 60)
            print("🔥 Your parser is BULLETPROOF! It can handle:")
            print("   ✅ Multi-language codebases")
            print("   ✅ Complex inheritance patterns")
            print("   ✅ Generic programming constructs")
            print("   ✅ Decorator and annotation patterns")
            print("   ✅ Async/await patterns")
            print("   ✅ Cross-file dependencies")
            print("   ✅ Advanced relationship detection")
            print("=" * 60)
            return 0
        else:
            print("\n" + "=" * 60)
            print("💥 MEGA VERIFICATION TEST SUITE: ❌ FAILED")
            print("=" * 60)
            print("Some advanced patterns were not detected correctly.")
            print("Check the detailed results above for specific issues.")
            print("=" * 60)
            return 1

    except Exception as e:
        print(f"\n❌ MEGA verification test crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)