import pytest
from hydra.core.override_parser.overrides_parser import OverridesParser
from src.command_aliases import graphs_overrides_from_alias

@pytest.mark.parametrize('path', [
    '/scratch/local.ssd/xsajko01/job_24546306[1].pbs-m1/work/repo/analysis_out/run',
    '/scratch.ssd/xsajko01/job_24546306[72].pbs-m1/a (copy),v2/run',
    '/tmp/report "quoted"/run',
    r'C:\reports\job[1]\run',
    'analysis_out/žluťoučký[1]/run',
])
@pytest.mark.parametrize('command,key', [('run', 'graphs.run_root')])
def test_graph_path_survives_actual_hydra_parser(path, command, key):
    overrides = graphs_overrides_from_alias([command, path])
    parsed = OverridesParser.create().parse_overrides(overrides)
    value = next(item.value() for item in parsed if item.key_or_group == key)
    assert value == path
