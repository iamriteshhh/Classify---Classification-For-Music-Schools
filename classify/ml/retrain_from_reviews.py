"""
retrain_from_reviews.py
=======================
Convenience CLI entry point for human-in-the-loop active learning.
Invokes classify.ml.active_learning.retrain_from_reviews().
Part of Phase 1 for CLASSIFY.
"""

from classify.ml.active_learning import retrain_from_reviews

if __name__ == "__main__":
    result = retrain_from_reviews()
    import json
    print(json.dumps(result, indent=2))
