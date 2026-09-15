from santapp_ruler.cli import build_parser, command_list_tasks, main


def test_list_tasks_prints_all_tasks(capsys) -> None:
    assert command_list_tasks() == 0
    output = capsys.readouterr().out
    assert "niah_single_1" in output
    assert "qa_2" in output
    assert output.count("\n") >= 15


def test_cli_rejects_removed_backend(capsys) -> None:
    try:
        main(["run", "--backends", "legacy"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("argparse should reject a removed backend")
    assert "invalid choice" in capsys.readouterr().err


def test_cli_accepts_both_santapp_parent_policies() -> None:
    args = build_parser().parse_args(
        ["run", "--backends", "santapp", "hierarchical"]
    )
    assert args.backends == ["santapp", "hierarchical"]
