# DetectiveQA research rules

- Do not invent data, results, citations, model responses, or completed actions. Missing results stay null or absent.
- Keep all new large data, models, environments, caches, logs, and temporary files inside this D-drive project.
- Use scripts/environment.ps1 before commands. Do not change global Python packages or user-wide cache settings.
- Only run development questions until the experimental protocol is frozen. Never tune on held-out results.
- Ordinary retrieval and answer prompts must not receive gold answers, reasoning, clue positions, or the answer-reveal paragraph.
- Keep novel variants and translations together. Preserve original raw files and log all exclusions.
- Do not redistribute source novels, secrets, or private billing information.
- No paid API call is part of environment preparation. Follow the recorded course requirements and budget before enabling paid runs.
- Record actual human contributions and AI assistance. Do not fabricate human ratings.
