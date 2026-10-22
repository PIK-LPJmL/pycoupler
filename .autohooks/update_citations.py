from contextlib import contextmanager
from pathlib import Path
from rich.progress import Progress
import os
import re
from autohooks.api import ReportProgress
from autohooks.api.git import (
    get_staged_status,
    stage_files,
    stash_unstaged_changes,
    exec_git,
)
from rich.prompt import Confirm

from packaging.version import Version
import sys
import yaml
import tomllib as toml

pyproject_path = Path("pyproject.toml")


has_stdin = False
if os.path.exists("/dev/tty"):
    sys.stdin = open("/dev/tty", "r")
    has_stdin = True


# https://github.com/Textualize/rich/issues/1535#issuecomment-1745297594
@contextmanager
def hide(progress: Progress):
    transient = progress.live.transient  # save the old value
    progress.live.transient = True
    progress.stop()
    progress.live.transient = transient  # restore the old value
    try:
        yield
    finally:
        # make space for the progress to use so it doesn't overwrite any previous lines
        print("\n" * (len(progress.tasks) - 2))
        progress.start()


def precommit(report_progress: ReportProgress, config=None, **kwargs):
    pyproject_changed = False
    for f in get_staged_status():
        if f.path == pyproject_path:
            pyproject_changed = True
            break
    if not pyproject_changed:
        return 0

    with stash_unstaged_changes():
        with pyproject_path.open("rb") as f:
            project_info = toml.load(f)
        with Path("CITATION.cff").open() as f:
            citations = yaml.load(f, Loader=yaml.Loader)

        changed_citations = False
        project_version = Version(project_info["project"]["version"])
        if Version(citations["version"]) != project_info:
            citations["version"] = str(project_version)
            changed_citations = True

            tags = exec_git("tag", "-l", "v*", "--sort", "v:refname").split()
            last_tag = tags[-1]
            authors_since_last_tag = exec_git(
                "shortlog", "--summary", "--email", f"{last_tag}..HEAD"
            ).splitlines()

            def find_cited_author(name=None, email=None):
                for i, a in enumerate(citations["authors"]):
                    if name in [
                        f"{a.get('given-names', '')} {a.get('family-names', '')}",
                        f"{a.get('family-names', '')} {a.get('given-names', '')}",
                    ]:
                        return i
                    elif email == a.get("email", ""):
                        return i
                return None

            with hide(report_progress._progress):
                # git hooks can be passed arguments via stdin, which messes up the prompt here,
                # so we flush the input
                author_regex = re.compile("(.*) <(.*)>")
                for author in authors_since_last_tag:
                    add_author = False
                    author = author.split("\t")[1]
                    match = author_regex.search(author)
                    if match:
                        name = match.group(1)
                        email = match.group(2)

                        author_index = find_cited_author(name=name)

                        if author_index is not None:
                            if citations["authors"][author_index]["email"] != email:
                                # Names match, but emails don’t
                                add_author = Confirm.ask(
                                    f"Add {name} <{email}> to CITATION.cff? There already is another author with the same name.",
                                    default=False,
                                )
                        else:
                            author_index = find_cited_author(email=email)
                            if author_index is not None:
                                # emails match, but names don’t
                                add_author = Confirm.ask(
                                    f"Add {name} <{email}> to CITATION.cff? There already is another author with the same email.",
                                    default=False,
                                )
                            else:
                                # completely unknown author
                                add_author = Confirm.ask(
                                    f"Add {name} <{email}> to CITATION.cff?",
                                    default=False,
                                )

                        if add_author:
                            citations["authors"].append(
                                {
                                    "given-names": name.split()[0],
                                    "family-names": " ".join(name.split()[1:]),
                                    "email": email,
                                }
                            )
                            changed_citations = True

        if changed_citations:
            with Path("CITATION.cff").open("w") as f:
                yaml.dump(citations, f)
            stage_files([Path("CITATION.cff")])
