# How to contribute to stac-model

### Project setup

1. If you don't have `Poetry` installed run:

```bash
make poetry-install
```

> This installs Poetry as a [standalone application][fs1]. If you prefer, you can simply install it inside your virtual environment.

2. Initialize project dependencies with poetry and install `pre-commit` hooks:

```bash
make install
make pre-commit-install
```

You're then ready to run and test your contributions.

To activate your `virtualenv` run `poetry shell`.

Want to know more about Poetry? Check [its documentation][fs2].

Poetry's [commands][fs3] let you easily make descriptive python environments and run commands in those environments, like:

- `poetry add numpy@latest`
- `poetry run pytest`
- `poetry publish --build`

etc.

3. Run the codestyle and other checks:

```bash
make codestyle
```

Many checks are configured for this project. Command `make check-codestyle` will run ruff for linting and autoformatting. `make lint` will just run linting. `make check-safety` will look at the security of your code.

Command `make lint-all` applies all checks.


4. Run `pytest` with

```bash
make test
```


5. Upload your changes to your fork, then make a PR from there to the main repo:

```bash
git checkout -b your-branch
git add .
git commit -m ":tada: Initial commit"
git remote add origin https://github.com/your-fork/stac-model.git
git push -u origin your-branch
```

### Building and releasing stac-model

Building a new version of `stac-model` contains steps:

- Bump the version with `poetry version <version>`. You can pass the new version explicitly, or a rule such as `major`, `minor`, or `patch`. For more details, refer to the [Semantic Versions][fs4] standard;
- Make a commit to `GitHub`;
- Create a `GitHub release`;
- And... publish :slight_smile: `poetry publish --build`

### Before submitting

Before submitting your code please do the following steps:

1. Add any changes you want
1. Add tests for the new changes
1. Edit documentation if you have changed something significant
1. Run `make codestyle` to format your changes.
1. Run `make lint-all` to ensure that types, security and docstrings are okay.

## Other help

You can contribute by spreading a word about this library.
It would also be a huge contribution to write
a short article on how you are using this project.
You can also share how the ML Model extension does or does
not serve your needs with us in the Github Discussions or raise
Issues for bugs.

[fs1]: https://github.com/python-poetry/install.python-poetry.org
[fs2]: https://python-poetry.org/docs/
[fs3]: https://python-poetry.org/docs/cli/#commands
[fs4]: https://semver.org/

[li2]: http://www.pydocstyle.org/en/stable/
[li3]: https://github.com/jsh9/pydoclint
