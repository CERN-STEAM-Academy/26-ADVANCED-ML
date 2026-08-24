"""Unit tests for the solutions-to-student notebook build tool.

The student notebooks are build artefacts, so this tool is the only thing standing between an
authoring slip and a room full of people looking at a notebook that either gives the answer
away or refuses to run. Three properties therefore carry most of the weight here:

* nothing outside a marked block may change, in code or in prose;
* a hole must land at the marker's own indentation, so the stripped cell still parses;
* an unbalanced marker must fail the build rather than silently truncate an exercise.

Every fixture below is built from plain dictionaries in this file. The tests must not depend
on a notebook existing on disk, because the notebooks are exactly the thing this tool is used
to produce.
"""

import json

import pytest

from tools import make_student
from tools.make_student import (
    DEFAULT_HINT,
    MarkerError,
    build,
    count_markers,
    default_pairs,
    main,
    strip_code,
    strip_markdown,
    strip_notebook,
    strip_python_source,
)

# --- fixture helpers -------------------------------------------------------------
#
# nbformat allows a cell source to be either a single string or a list of lines, and both
# occur in the wild (Jupyter writes lists, hand-built notebooks often carry strings). The
# helpers keep both shapes available so the tests can exercise each of them.


def code_cell(source, outputs=None, execution_count=None, metadata=None):
    return {
        "cell_type": "code",
        "source": source,
        "outputs": [] if outputs is None else outputs,
        "execution_count": execution_count,
        "metadata": {} if metadata is None else metadata,
    }


def markdown_cell(source, metadata=None):
    return {
        "cell_type": "markdown",
        "source": source,
        "metadata": {} if metadata is None else metadata,
    }


def notebook(*cells):
    return {
        "cells": list(cells),
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def cell_text(cell):
    """A cell's source as one string, whichever of the two shapes it is stored in."""
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


def strip_code_text(text, where="notebook cell 0 (code)"):
    """Run ``strip_code`` over a block of text and hand back text, for readable assertions."""
    new_lines, removed = strip_code(text.splitlines(keepends=True), where)
    return "".join(new_lines), removed


def strip_markdown_text(text, where="notebook cell 0 (markdown)"):
    new_lines, removed = strip_markdown(text.splitlines(keepends=True), where)
    return "".join(new_lines), removed


def write_notebook(path, nb):
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return str(path)


# A hole three levels deep. Indentation is the thing most likely to break, and it breaks
# invisibly: the student notebook still looks fine and only fails when someone runs it.
NESTED_SOLUTION = """def policy_evaluation(env, policy, gamma=0.99):
    values = np.zeros(env.n_states)
    for sweep in range(100):
        for state in range(env.n_states):
            # TODO(hint): apply one Bellman backup to this state
            # BEGIN SOLUTION
            total = 0.0
            for action, probability in enumerate(policy[state]):
                total += probability * env.expected_return(state, action, values, gamma)
            values[state] = total
            # END SOLUTION
    return values
"""


# --- code cells: the happy path --------------------------------------------------


def test_hole_becomes_a_plain_todo_and_a_raise():
    text = """# TODO(hint): compute the group-normalised advantage
# BEGIN SOLUTION
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
# END SOLUTION
"""
    student, removed = strip_code_text(text)
    assert removed == 1
    assert student == "# TODO: compute the group-normalised advantage\nraise NotImplementedError\n"


def test_solution_body_leaves_no_trace():
    student, _ = strip_code_text(NESTED_SOLUTION)
    for leaked in ("total", "expected_return", "probability", "BEGIN SOLUTION", "END SOLUTION"):
        assert leaked not in student


def test_hint_text_is_preserved_apart_from_dropping_the_hint_marker():
    # The hint is prose the author wrote for the student; punctuation, backticks and LaTeX in
    # it must survive untouched, since only the "(hint)" part carries meaning for the build.
    text = """# TODO(hint): return $\\sum_a \\pi(a|s) q(s, a)$, using `action_values` - do not loop twice
# BEGIN SOLUTION
return float(policy[state] @ action_values(env, values, state, gamma))
# END SOLUTION
"""
    student, _ = strip_code_text(text)
    first_line = student.splitlines()[0]
    assert first_line == "# TODO: return $\\sum_a \\pi(a|s) q(s, a)$, using `action_values` - do not loop twice"


def test_indentation_of_the_marker_is_reproduced_on_both_replacement_lines():
    student, removed = strip_code_text(NESTED_SOLUTION)
    assert removed == 1
    assert "            # TODO: apply one Bellman backup to this state\n" in student
    assert "            raise NotImplementedError\n" in student
    # Nothing outside the block moved: the wrapper is byte for byte what it was.
    assert student.startswith("def policy_evaluation(env, policy, gamma=0.99):\n    values = np.zeros(env.n_states)\n")
    assert student.endswith("    return values\n")


def test_stripped_cell_is_still_valid_python():
    # The indentation assertion above is only meaningful because of this one: a raise at the
    # wrong indentation turns an exercise into an IndentationError at the top of the cell.
    student, _ = strip_code_text(NESTED_SOLUTION)
    compile(student, "<student cell>", "exec")


def test_multiple_holes_in_one_cell_are_each_replaced():
    text = """# TODO(hint): initialise the value table
# BEGIN SOLUTION
values = np.zeros(env.n_states)
# END SOLUTION

for sweep in range(sweeps):
    # TODO(hint): one synchronous sweep of the Bellman optimality backup
    # BEGIN SOLUTION
    values = np.max(action_values(env, values, gamma), axis=1)
    # END SOLUTION

# TODO(hint): read the greedy policy off the converged values
# BEGIN SOLUTION
policy = greedy_policy(env, values, gamma)
# END SOLUTION
"""
    student, removed = strip_code_text(text)
    assert removed == 3
    assert student == """# TODO: initialise the value table
raise NotImplementedError

for sweep in range(sweeps):
    # TODO: one synchronous sweep of the Bellman optimality backup
    raise NotImplementedError

# TODO: read the greedy policy off the converged values
raise NotImplementedError
"""


def test_multi_line_hint_keeps_its_continuation_comments():
    # Only the TODO(hint) line is rewritten. The comment lines below it are part of the hint
    # the author wrote and belong in the student notebook exactly as typed.
    text = """    # TODO(hint): fill in the one-step lookahead
    #   the sum runs over next states, weighted by the transition probability
    #   remember the terminal state contributes no bootstrap term
    # BEGIN SOLUTION
    values[state] = backup(state)
    # END SOLUTION
"""
    student, removed = strip_code_text(text)
    assert removed == 1
    assert student == """    # TODO: fill in the one-step lookahead
    #   the sum runs over next states, weighted by the transition probability
    #   remember the terminal state contributes no bootstrap term
    raise NotImplementedError
"""


def test_hole_with_no_hint_still_gets_a_todo(capsys):
    text = """def dqn_target(batch, target_net, gamma):
    # BEGIN SOLUTION
    return batch.rewards + gamma * target_net(batch.next_states).max(1).values
    # END SOLUTION
"""
    student, removed = strip_code_text(text)
    assert removed == 1
    assert student == f"""def dqn_target(batch, target_net, gamma):
    # TODO: {DEFAULT_HINT}
    raise NotImplementedError
"""
    compile(student, "<student cell>", "exec")
    # A hole with no hint is legal but almost always an oversight, so it is reported.
    assert "no TODO(hint)" in capsys.readouterr().err


def test_hole_with_no_hint_keeps_ordinary_comments_above_it():
    text = """# the target network is deliberately stale
# BEGIN SOLUTION
target = target_net(next_states)
# END SOLUTION
"""
    student, _ = strip_code_text(text)
    assert student == f"""# the target network is deliberately stale
# TODO: {DEFAULT_HINT}
raise NotImplementedError
"""


def test_hole_on_the_very_first_line_of_a_cell_has_nothing_to_scan_back_over():
    text = "    # BEGIN SOLUTION\n    values[state] = backup\n    # END SOLUTION\n"
    student, removed = strip_code_text(text)
    assert removed == 1
    assert student == f"    # TODO: {DEFAULT_HINT}\n    raise NotImplementedError\n"


def test_code_outside_the_markers_is_untouched():
    text = """import numpy as np

GAMMA = 0.99  # discount, chosen so the horizon is about 100 steps


def epsilon_greedy(q_values, epsilon, rng):
    \"\"\"Pick an action, exploring with probability epsilon.\"\"\"
    # TODO(hint): return a uniformly random action with probability epsilon
    # BEGIN SOLUTION
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    return int(np.argmax(q_values))
    # END SOLUTION


print("module loaded")
"""
    student, _ = strip_code_text(text)
    head, _, tail = student.partition("    # TODO: return a uniformly random action with probability epsilon\n")
    assert head == text.split("    # TODO(hint):")[0]
    assert tail == "    raise NotImplementedError\n\n\nprint(\"module loaded\")\n"


def test_marker_spacing_variants_are_recognised():
    text = """#TODO(hint):no space anywhere
#BEGIN SOLUTION
x = 1
#END SOLUTION
#   TODO(hint):   padded hint
#   BEGIN SOLUTION\t
x = 2
#   END SOLUTION   
"""
    student, removed = strip_code_text(text)
    assert removed == 2
    # Indentation comes from what precedes the "#", not from the padding inside the comment,
    # so both replacements sit hard against the left margin here.
    assert student == """# TODO: no space anywhere
raise NotImplementedError
# TODO: padded hint
raise NotImplementedError
"""


def test_lines_that_merely_look_like_markers_are_not_markers():
    text = """## BEGIN SOLUTION
# BEGIN SOLUTION for part 2
# begin solution
banner = "# BEGIN SOLUTION"
# END SOLUTION of part 2
"""
    student, removed = strip_code_text(text)
    assert removed == 0
    assert student == text


def test_code_without_markers_is_returned_unchanged():
    text = "for step in range(10):\n    print(step)\n"
    student, removed = strip_code_text(text)
    assert removed == 0
    assert student == text


# --- code cells: unbalanced markers ----------------------------------------------


def test_end_without_begin_is_an_error():
    text = "x = 1\n# END SOLUTION\n"
    with pytest.raises(MarkerError) as caught:
        strip_code_text(text, where="01_classics cell 4 (code)")
    message = str(caught.value)
    assert "01_classics cell 4 (code)" in message
    assert "line 2" in message


def test_begin_without_end_is_an_error():
    text = "# TODO(hint): fill this in\n# BEGIN SOLUTION\nx = 1\n"
    with pytest.raises(MarkerError) as caught:
        strip_code_text(text, where="01_classics cell 4 (code)")
    assert "never closed" in str(caught.value)
    assert "01_classics cell 4 (code)" in str(caught.value)


def test_unclosed_begin_is_reported_at_its_own_line():
    # The author has to find the marker in a cell that may be a hundred lines long, so the
    # number in the message has to be the line the BEGIN is on, not wherever the scan gave up.
    text = "import numpy as np\n# BEGIN SOLUTION\n" + "".join(f"line {i}\n" for i in range(3, 40))
    with pytest.raises(MarkerError) as caught:
        strip_code_text(text)
    assert "at line 2 was never closed" in str(caught.value)


def test_nested_begin_is_an_error():
    text = """# BEGIN SOLUTION
x = 1
# BEGIN SOLUTION
y = 2
# END SOLUTION
# END SOLUTION
"""
    with pytest.raises(MarkerError) as caught:
        strip_code_text(text)
    assert "nested" in str(caught.value)
    assert "line 3" in str(caught.value)


def test_a_second_unclosed_block_after_a_good_one_is_still_caught():
    text = """# BEGIN SOLUTION
x = 1
# END SOLUTION
# BEGIN SOLUTION
y = 2
"""
    with pytest.raises(MarkerError):
        strip_code_text(text)


# --- markdown cells --------------------------------------------------------------


def test_markdown_hole_becomes_an_answer_placeholder():
    text = """<!-- TODO(hint): which leg of the deadly triad did each config break? -->
<!-- BEGIN SOLUTION -->
CONFIG_A removes the target network, so the bootstrap target chases the policy network.
CONFIG_B shrinks the buffer until consecutive samples are correlated.
<!-- END SOLUTION -->
"""
    student, removed = strip_markdown_text(text)
    assert removed == 1
    assert student == """> **Your answer** - which leg of the deadly triad did each config break?
>
> _(replace this line)_
"""


def test_markdown_hole_without_a_hint_gets_a_bare_placeholder():
    text = "<!-- BEGIN SOLUTION -->\nBecause the target moves every step.\n<!-- END SOLUTION -->\n"
    student, removed = strip_markdown_text(text)
    assert removed == 1
    assert student == "> **Your answer**\n>\n> _(replace this line)_\n"


def test_markdown_narrative_and_latex_are_preserved_byte_for_byte():
    text = """## 2.3 Why the target network exists

Q-learning bootstraps from its own estimate:

$$ y_t = r_t + \\gamma \\max_{a'} Q_{\\theta^-}(s_{t+1}, a') $$

If $\\theta^- = \\theta$, the regression target moves with every gradient step.

<!-- TODO(hint): what happens to $y_t$ when the target network is removed? -->
<!-- BEGIN SOLUTION -->
The target chases the prediction and the two diverge together.
<!-- END SOLUTION -->

The next section measures exactly that.
"""
    student, removed = strip_markdown_text(text)
    assert removed == 1
    prologue, _, epilogue = student.partition("> **Your answer** - what happens to $y_t$ when the target network is removed?\n")
    assert prologue == text.split("<!-- TODO(hint):")[0]
    assert epilogue == ">\n> _(replace this line)_\n\nThe next section measures exactly that.\n"


def test_markdown_that_merely_mentions_solutions_is_untouched():
    text = """## Exercise 3: the deadly triad

The solution below is deliberately unstable. Everything between a `<!-- BEGIN SOLUTION -->`
line and the matching `<!-- END SOLUTION -->` line is removed by `tools/make_student.py`,
which is why the solutions notebook and the student notebook read as the same document.

<!-- an ordinary HTML comment that mentions END SOLUTION in passing -->

Solutions are published after the session.
"""
    student, removed = strip_markdown_text(text)
    assert removed == 0
    assert student == text


def test_multiple_markdown_holes_in_one_cell():
    text = """First question.

<!-- TODO(hint): name the leg of the triad -->
<!-- BEGIN SOLUTION -->
Bootstrapping.
<!-- END SOLUTION -->

Second question.

<!-- BEGIN SOLUTION -->
Off-policy replay.
<!-- END SOLUTION -->
"""
    student, removed = strip_markdown_text(text)
    assert removed == 2
    assert student == """First question.

> **Your answer** - name the leg of the triad
>
> _(replace this line)_

Second question.

> **Your answer**
>
> _(replace this line)_
"""


def test_markdown_end_without_begin_is_an_error():
    with pytest.raises(MarkerError) as caught:
        strip_markdown_text("text\n<!-- END SOLUTION -->\n", where="02_grpo cell 9 (markdown)")
    assert "02_grpo cell 9 (markdown)" in str(caught.value)


def test_markdown_begin_without_end_is_an_error():
    with pytest.raises(MarkerError) as caught:
        strip_markdown_text("<!-- BEGIN SOLUTION -->\nan answer\n")
    assert "never closed" in str(caught.value)


def test_unclosed_markdown_begin_is_reported_at_its_own_line():
    text = "Some prose.\n\n<!-- BEGIN SOLUTION -->\n" + "".join(f"answer line {i}\n" for i in range(4, 20))
    with pytest.raises(MarkerError) as caught:
        strip_markdown_text(text)
    assert "at line 3 was never closed" in str(caught.value)


def test_markdown_nested_begin_is_an_error():
    text = "<!-- BEGIN SOLUTION -->\n<!-- BEGIN SOLUTION -->\nx\n<!-- END SOLUTION -->\n<!-- END SOLUTION -->\n"
    with pytest.raises(MarkerError) as caught:
        strip_markdown_text(text)
    assert "nested" in str(caught.value)


def test_markdown_without_markers_is_returned_unchanged():
    text = "# Title\n\nSome prose with a $\\LaTeX$ formula.\n"
    student, removed = strip_markdown_text(text)
    assert removed == 0
    assert student == text


# --- whole notebooks -------------------------------------------------------------


def solutions_notebook():
    """A miniature but representative solutions notebook, in the two source shapes."""
    return notebook(
        markdown_cell("# Notebook 1: classics\n\nPolicy evaluation, then value iteration.\n"),
        code_cell(
            ["import numpy as np\n", "\n", "print(np.__version__)\n"],
            outputs=[{"output_type": "stream", "name": "stdout", "text": ["1.24.4\n"]}],
            execution_count=1,
            metadata={"execution": {"iopub.execute_input": "2026-08-24T10:00:00Z"}},
        ),
        code_cell(
            NESTED_SOLUTION,
            outputs=[{"output_type": "execute_result", "data": {"text/plain": ["array([0., 0.])"]}}],
            execution_count=2,
        ),
        markdown_cell(
            "## Discussion\n"
            "\n"
            "<!-- TODO(hint): why does the sweep converge? -->\n"
            "<!-- BEGIN SOLUTION -->\n"
            "The Bellman operator is a contraction.\n"
            "<!-- END SOLUTION -->\n"
        ),
        {"cell_type": "raw", "source": "raw cells are left alone\n", "metadata": {}},
    )


def test_strip_notebook_reports_what_it_did():
    student, stats = strip_notebook(solutions_notebook(), name="01_classics_solutions.ipynb")
    assert stats == {"cells": 5, "code_blocks": 1, "markdown_blocks": 1}
    assert count_markers(student) == 0


def test_strip_notebook_clears_outputs_and_execution_state():
    student, _ = strip_notebook(solutions_notebook())
    for cell in student["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            assert "execution" not in cell["metadata"]


def test_strip_notebook_does_not_mutate_its_input():
    original = solutions_notebook()
    before = json.dumps(original, sort_keys=True)
    strip_notebook(original)
    assert json.dumps(original, sort_keys=True) == before


def test_strip_notebook_leaves_raw_cells_alone():
    student, _ = strip_notebook(solutions_notebook())
    assert student["cells"][4] == {"cell_type": "raw", "source": "raw cells are left alone\n", "metadata": {}}


def test_strip_notebook_keeps_top_level_metadata():
    original = solutions_notebook()
    student, _ = strip_notebook(original)
    assert student["metadata"] == original["metadata"]
    assert student["nbformat"] == 4
    assert student["nbformat_minor"] == 5


def test_notebook_without_markers_only_loses_its_outputs():
    original = notebook(
        markdown_cell("## A section\n\nWith $\\gamma = 0.99$ and no holes at all.\n"),
        code_cell(
            "values = np.zeros(4)\nprint(values.shape)\n",
            outputs=[{"output_type": "stream", "name": "stdout", "text": ["(4,)\n"]}],
            execution_count=7,
        ),
    )
    student, stats = strip_notebook(original)

    assert stats["code_blocks"] == 0 and stats["markdown_blocks"] == 0
    # The source may be re-normalised from a string to a list of lines, which nbformat treats
    # as the same document, so compare the text rather than the container.
    for new_cell, old_cell in zip(student["cells"], original["cells"]):
        assert cell_text(new_cell) == cell_text(old_cell)
    assert student["cells"][1]["outputs"] == []
    assert student["cells"][1]["execution_count"] is None


@pytest.mark.parametrize(
    "source",
    [
        "# TODO(hint): do it\n# BEGIN SOLUTION\nx = 1\n# END SOLUTION\n",
        ["# TODO(hint): do it\n", "# BEGIN SOLUTION\n", "x = 1\n", "# END SOLUTION\n"],
    ],
    ids=["string-source", "list-source"],
)
def test_code_source_may_be_a_string_or_a_list_of_lines(source):
    student, stats = strip_notebook(notebook(code_cell(source)))
    assert stats["code_blocks"] == 1
    assert cell_text(student["cells"][0]) == "# TODO: do it\nraise NotImplementedError\n"


@pytest.mark.parametrize(
    "source",
    [
        "<!-- TODO(hint): why? -->\n<!-- BEGIN SOLUTION -->\nBecause.\n<!-- END SOLUTION -->\n",
        ["<!-- TODO(hint): why? -->\n", "<!-- BEGIN SOLUTION -->\n", "Because.\n", "<!-- END SOLUTION -->\n"],
    ],
    ids=["string-source", "list-source"],
)
def test_markdown_source_may_be_a_string_or_a_list_of_lines(source):
    student, stats = strip_notebook(notebook(markdown_cell(source)))
    assert stats["markdown_blocks"] == 1
    assert cell_text(student["cells"][0]) == "> **Your answer** - why?\n>\n> _(replace this line)_\n"


def test_a_missing_source_key_is_tolerated():
    cell = {"cell_type": "code", "metadata": {}}
    student, stats = strip_notebook(notebook(cell))
    assert stats["code_blocks"] == 0
    assert cell_text(student["cells"][0]) == ""


def test_strip_notebook_is_idempotent():
    once, first_stats = strip_notebook(solutions_notebook())
    twice, second_stats = strip_notebook(once)
    assert twice == once
    assert second_stats["code_blocks"] == 0
    assert second_stats["markdown_blocks"] == 0
    assert first_stats["code_blocks"] == 1


def test_count_markers_sees_both_kinds_of_block():
    assert count_markers(solutions_notebook()) == 2
    assert count_markers(notebook()) == 0


# --- build() and the command line ------------------------------------------------


def test_build_writes_a_notebook_with_no_markers_left(tmp_path):
    source = write_notebook(tmp_path / "01_classics_solutions.ipynb", solutions_notebook())
    dest = str(tmp_path / "01_classics_student.ipynb")

    stats = build(source, dest, verbose=False)
    assert stats["code_blocks"] == 1 and stats["markdown_blocks"] == 1

    raw = (tmp_path / "01_classics_student.ipynb").read_text()
    assert raw.endswith("\n")
    assert "BEGIN SOLUTION" not in raw and "END SOLUTION" not in raw
    assert "expected_return" not in raw  # the solution body itself is gone
    student = json.loads(raw)
    assert count_markers(student) == 0
    assert cell_text(student["cells"][0]) == cell_text(solutions_notebook()["cells"][0])


def test_build_is_idempotent_on_its_own_output(tmp_path):
    source = write_notebook(tmp_path / "01_classics_solutions.ipynb", solutions_notebook())
    first = tmp_path / "student_once.ipynb"
    second = tmp_path / "student_twice.ipynb"

    build(source, str(first), verbose=False)
    stats = build(str(first), str(second), verbose=False)

    assert stats["code_blocks"] == 0 and stats["markdown_blocks"] == 0
    assert second.read_text() == first.read_text()


def test_build_creates_the_destination_directory(tmp_path):
    source = write_notebook(tmp_path / "01_classics_solutions.ipynb", solutions_notebook())
    dest = tmp_path / "generated" / "notebooks" / "01_classics_student.ipynb"
    build(source, str(dest), verbose=False)
    assert dest.exists()


def test_build_refuses_a_notebook_whose_markers_survived(tmp_path):
    # A code-style marker sitting in a markdown cell is never stripped, because markdown holes
    # use the HTML-comment form. The post-build count is the safety net that catches it before
    # the leaked solution reaches a student.
    nb = notebook(markdown_cell("Authoring slip below.\n\n# BEGIN SOLUTION\nthe answer\n# END SOLUTION\n"))
    source = write_notebook(tmp_path / "x_solutions.ipynb", nb)
    with pytest.raises(MarkerError) as caught:
        build(source, str(tmp_path / "x_student.ipynb"), verbose=False)
    assert "survived" in str(caught.value)


def test_build_reports_the_pair_when_verbose(tmp_path, capsys):
    source = write_notebook(tmp_path / "01_classics_solutions.ipynb", solutions_notebook())
    build(source, str(tmp_path / "01_classics_student.ipynb"), verbose=True)
    out = capsys.readouterr().out
    assert "1 code holes" in out and "1 prose holes" in out


def test_default_pairs_maps_solutions_to_student_notebooks(tmp_path):
    for name in ("02_grpo_solutions.ipynb", "01_classics_solutions.ipynb", "01_classics_student.ipynb", "notes.md"):
        (tmp_path / name).write_text("{}")

    pairs = default_pairs(str(tmp_path))

    assert pairs == [
        (str(tmp_path / "01_classics_solutions.ipynb"), str(tmp_path / "01_classics_student.ipynb")),
        (str(tmp_path / "02_grpo_solutions.ipynb"), str(tmp_path / "02_grpo_student.ipynb")),
    ]


def test_main_builds_every_solutions_notebook_in_the_directory(tmp_path):
    write_notebook(tmp_path / "01_classics_solutions.ipynb", solutions_notebook())
    write_notebook(tmp_path / "02_grpo_solutions.ipynb", solutions_notebook())

    assert main(["--notebooks-dir", str(tmp_path), "--quiet"]) == 0

    for name in ("01_classics_student.ipynb", "02_grpo_student.ipynb"):
        student = json.loads((tmp_path / name).read_text())
        assert count_markers(student) == 0


def test_main_accepts_an_explicit_source_and_destination(tmp_path):
    source = write_notebook(tmp_path / "01_classics_solutions.ipynb", solutions_notebook())
    dest = tmp_path / "elsewhere" / "student.ipynb"
    assert main([source, str(dest), "--quiet"]) == 0
    assert dest.exists()


def test_main_rejects_a_source_without_a_destination(tmp_path):
    source = write_notebook(tmp_path / "01_classics_solutions.ipynb", solutions_notebook())
    with pytest.raises(SystemExit):
        main([source, "--quiet"])


def test_main_returns_two_and_explains_itself_on_unbalanced_markers(tmp_path, capsys):
    broken = notebook(code_cell("# BEGIN SOLUTION\nx = 1\n"))
    source = write_notebook(tmp_path / "01_classics_solutions.ipynb", broken)

    assert main([source, str(tmp_path / "01_classics_student.ipynb"), "--quiet"]) == 2

    err = capsys.readouterr().err
    assert "ERROR" in err and "never closed" in err
    assert not (tmp_path / "01_classics_student.ipynb").exists()


def test_main_returns_one_when_there_is_nothing_to_build(tmp_path, capsys):
    assert main(["--notebooks-dir", str(tmp_path), "--quiet"]) == 1
    assert "no *_solutions.ipynb found" in capsys.readouterr().err


# --- plain .py files -------------------------------------------------------------


PY_MODULE = '''"""Reference implementations, with holes for the student build."""


class Learner:
    def update(self, batch):
        """One gradient step."""
        # TODO(hint): compute the TD error and take one optimiser step
        # BEGIN SOLUTION
        loss = self.loss_fn(self.q(batch.states), batch.targets)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return float(loss)
        # END SOLUTION


def helper(x):
    return x + 1
'''


def test_strip_python_source_on_a_module(tmp_path):
    path = tmp_path / "learner.py"
    path.write_text(PY_MODULE)

    stripped, removed = strip_python_source(path.read_text(), name="learner.py")

    assert removed == 1
    assert "        # TODO: compute the TD error and take one optimiser step\n" in stripped
    assert "        raise NotImplementedError\n" in stripped
    assert "optimiser.step" not in stripped
    assert stripped.endswith("def helper(x):\n    return x + 1\n")
    compile(stripped, str(path), "exec")


def test_strip_python_source_is_idempotent(tmp_path):
    path = tmp_path / "learner.py"
    path.write_text(PY_MODULE)

    once, first_removed = strip_python_source(path.read_text(), name="learner.py")
    path.write_text(once)
    twice, second_removed = strip_python_source(path.read_text(), name="learner.py")

    assert first_removed == 1
    assert second_removed == 0
    assert twice == once


def test_strip_python_source_without_markers_returns_the_text_verbatim():
    text = "import numpy as np\n\n\ndef f(x):\n    return np.tanh(x)\n"
    stripped, removed = strip_python_source(text)
    assert removed == 0
    assert stripped == text


def test_strip_python_source_reports_unbalanced_markers():
    with pytest.raises(MarkerError) as caught:
        strip_python_source("# END SOLUTION\n", name="rewards.py")
    assert "rewards.py" in str(caught.value)


def test_module_exposes_the_documented_names():
    # The notebooks and the Makefile import these by name; renaming one silently breaks the
    # build, so pin the public surface here.
    for name in ("strip_code", "strip_markdown", "strip_notebook", "strip_python_source",
                 "count_markers", "build", "default_pairs", "main", "MarkerError"):
        assert hasattr(make_student, name)
    assert issubclass(MarkerError, RuntimeError)
