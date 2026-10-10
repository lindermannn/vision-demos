# Pre-publication audit

## Sequence

1. Run `scripts/validate_publication.py`.
2. Manually review every potential secret or private-path match.
3. Scan Git history if it exists.
4. Confirm that private exercise data and unlicensed weights are excluded.
5. When executable code is added, run its tests from a clean installation.
6. Review the README, license, and security contact.
7. Create a portfolio tag only after review.

## Blocking criteria

A token, video, database, large checkpoint, or confidential document blocks pushing until it is removed and its absence from history is verified.
