# How to contribute to MLM specification or `stac-model`

## Project setup

1. If you don't have `uv` installed run:

   ```bash
   make setup
   ```

   This installs `uv` as a [standalone application][uv-install]. <br>
   For more details, see also the [`uv` documentation][uv-docs].

2. Initialize project dependencies with `uv` and install `pre-commit` hooks:

   ```bash
   make install-dev
   make pre-commit-install
   ```

   This will install project dependencies into the currently active environment. If you would like to
   use `uv`'s default behavior of managing a project-scoped environment, use `uv` commands directly to
   install dependencies. `uv sync` will install dependencies and dev dependencies in `.venv` and update the `uv.lock`.

## PR submission

Before submitting your code please do the following steps:

1. Add any changes you want

2. Add tests for the new changes

3. Edit documentation if you have changed something significant

   You're then ready to run and test your contributions.

4. Run linting checks:

   ```bash
   make lint-all
   ```

5. Run `tests` (including your new ones) with

   ```bash
   make test
   ```

6. Upload your changes to your fork, then make a PR from there to the main repo:

   ```bash
   git checkout -b your-branch
   git add .
   git commit -m ":tada: Initial commit"
   git remote add origin https://github.com/your-fork/mlm-extension.git
   git push -u origin your-branch
   ```

## Building and releasing

<!-- lint disable no-undefined-references -->

> [!WARNING]
> There are multiple types of releases for this repository: <br>
>
> 1. Release for MLM specification (usually, this should include one for `stac-model` as well to support it)
> 2. Release for `stac-model` only

<!-- lint enable no-undefined-references -->

### Building a new version of MLM specification

- Checkout to the `main` branch by making sure the CI passed all previous tests.
- Bump the version with `bump-my-version bump --verbose <version-level>`.
  - Consider using `--dry-run` beforehand to inspect the changes.
  - The `<version-level>` should be one of `major`, `minor`, or `patch`. <br>
    Alternatively, the version can be set explicitly with `--new-version <version> patch`. <br>
    For more details, refer to the [Semantic Versions][semver] standard;
- Make a commit to `GitHub` and push the corresponding auto-generated `v{MAJOR}.{MINOR}.{PATCH}` tag.
- Validate that the CI validated everything once again.
- Create a `GitHub release` with the created tag.

<!-- lint disable no-undefined-references -->

> [!WARNING]
>
> - Ensure the "Set as the latest release" option is selected :heavy_check_mark:.
> - Ensure the diff ranges from the previous MLM version, and not an intermediate `stac-model` release.

<!-- lint enable no-undefined-references -->

### Building a new version of `stac-model`

- Apply any relevant changes and `CHANGELOG.md` entries in a PR that modifies `stac-model`.
- Bump the version with `bump-my-version bump --verbose <version-level> --config-file stac-model.bump.toml`.
  - You can pass the new version explicitly, or a rule such as `major`, `minor`, or `patch`. <br>
    For more details, refer to the [Semantic Versions][semver] standard;
- Once CI validation succeeded, merge the corresponding PR branch.
- Checkout to `main` branch that contains the freshly created merge commit.
- Push the tag `stac-model-v{MAJOR}.{MINOR}.{PATCH}`. The CI should auto-publish it to PyPI.
- Create a `GitHub release` (if not automatically drafted by the CI).

<!-- lint disable no-undefined-references -->

> [!WARNING]
>
> - Ensure the "Set as the latest release" option is deselected :x:.
> - Ensure the diff ranges from the previous release of `stac-model`, not an intermediate MLM release.

<!-- lint enable no-undefined-references -->

## Other help

You can contribute by spreading a word about this library.
It would also be a huge contribution to write
a short article on how you are using this project.
You can also share how the ML Model extension does or does
not serve your needs with us in the GitHub Discussions or raise
Issues for bugs.

[uv-install]: https://docs.astral.sh/uv/getting-started/installation/

[uv-docs]: https://docs.astral.sh/uv/

[semver]: https://semver.org/
