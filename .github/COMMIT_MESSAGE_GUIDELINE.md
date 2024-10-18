# Commit Message Guidelines

A good commit message should be descriptive and provide context about the changes made. This makes it easier to understand and review the changes in the future.

The specification used to write human-readable commits is [Conventional Commits](https://conventionalcommits.org).

Here are some guidelines for writing descriptive commit messages:

- Start with a short summary of the changes made in the commit.

- Use imperative mood for the summary, as if you're giving a command. For example, "Add feature" instead of "Added feature".

- Provide additional details in the commit message body, if necessary. This could include the reason for the change, the impact of the change, or any dependencies that were introduced or removed.

- Keep the message within 72 characters per line to ensure that it's easy to read in Git log output.

Examples of good commit messages:

- "Add authentication feature for user login"
- "Fix bug causing application to crash on startup"
- "Update documentation for API endpoints"

Remember, writing descriptive commit messages can save time and frustration in the future, and help others understand the changes made to the codebase.

# Allowed Commit Types

You must use one of the following types as the prefix for your commit messages:

### Commit Types

`feat`: adding a new feature to the project

```markdown
feat: add multi-image upload support
```

`fix`: fixing a bug or issue in the project

```markdown
fix: fix bug causing application to crash on startup
```

`docs`: updating documentation in the project

```markdown
docs: update documentation for api endpoints
```

`refactor`: making code changes that don't affect the behavior of the project, but improve its quality or maintainability

```markdown
refactor: remove unused code
```

`test`: adding or modifying tests for the project

```markdown
test: add tests for new feature
```

`chore`: making changes to the project that don't fit into any other category, such as updating dependencies or configuring the build system

```markdown
chore: update dependencies
```

`perf`: improving performance of the project

```markdown
perf: improve performance of image processing
```

`build`: making changes to the build system or dependencies of the project

```markdown
build: update dependencies
```

`ci`: making changes to the continuous integration (ci) system for the project

```markdown
ci: update ci configuration
```

`config`: making changes to configuration files for the project

```markdown
config: update configuration files
```

`init`: creating or initializing a new repository or project

```markdown
init: initialize project
```

`rename`: renaming files or directories within the project

```markdown
rename: rename files
```

## Gitlint for Commit Validation

`Gitlint` is used to enforce commit message rules, ensuring consistency in commit formats. `Gitlint` checks include:

    Character length: Ensures the subject line does not exceed 72 characters.
    Imperative tone: Validates that the summary uses an imperative mood.
    Proper formatting: Verifies a blank line between the summary and the body.

Example `Gitlint` Configuration

Add this configuration to `.pre-commit-config.yaml` for `Gitlint` integration:

```yaml
- repo: https://github.com/jorisroovers/gitlint
  rev: v0.19.1
  hooks:
    - id: gitlint
```

Run `pre-commit install` to activate Gitlint for commit validation.
