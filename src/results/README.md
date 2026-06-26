# Results

`results/` owns run layout, artifact IDs, manifest writes, and cache
invalidation.

Use `ArtifactWriter` to write artifacts and `RunArtifactReader` to read them.
Do not infer dependencies from filenames.

## Artifact Rules

- `config_hash` says whether the producer settings match.
- `content_hash` says whether the artifact payload changed.
- `inputs` store upstream artifact IDs and content hashes.
- Rewriting an artifact removes stale descendants whose input hashes no longer
  match.
- `ArtifactWriter.write()` holds the manifest lock while checking current state,
  writing the file, updating descendants, and writing the manifest.

If a new artifact type adds generated side files, teach stale cleanup how to
remove them.
