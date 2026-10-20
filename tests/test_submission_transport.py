"""Runtime contracts use inert subprocesses and temporary byte fixtures only."""
import hashlib
from dataclasses import replace
import json
import os
import subprocess
import sys
import time

import pytest

from submission_runtime.assets import verify_assets
from submission_runtime.journal import JournalError, SidecarJournal
from submission_runtime.transport import BridgeConfig, BridgeConflictError, BridgeError, ProcessBridge


WORKER = r'''
import sys,time
from submission_runtime.bridges.protocol import serve
class Worker:
    def handle(self, operation, payload, session_id, request_id):
        if operation == 'hello':
            return {'runtime':'synthetic-1','capabilities':['speech'],'family':'synthetic'}
        if operation == 'wait':
            time.sleep(3)
        if operation == 'effect':
            with open(payload['path'],'a') as stream:
                stream.write('effect\\n')
        if operation == 'fail':
            raise ValueError('synthetic rejected command')
        return {'value':payload.get('value'), 'session':session_id, 'request':request_id}
    def close(self):
        pass
serve(Worker())
'''


def config(**changes):
    values = dict(name='synthetic', command=[sys.executable, '-u', '-c', WORKER],
                  runtime='synthetic-1', capabilities=['speech'], timeout_seconds=1)
    values.update(changes)
    return BridgeConfig(**values)


def test_configuration_uses_argv_and_rejects_unbounded_or_unknown_fields():
    assert BridgeConfig.from_dict('x', {'command':['python','worker.py'], 'runtime':'r',
                                      'capabilities':['speech']}).name == 'x'
    assert BridgeConfig.from_dict({'name':'x','command':['python'], 'runtime':'r',
                                   'capabilities':[]}).command == ('python',)
    for bad in ({'command':'python worker.py'}, {'timeout_seconds':float('nan')},
                {'timeout_seconds':0}, {'timeout_seconds':1000}, {'max_message_bytes':10}):
        with pytest.raises(ValueError):
            config(**bad)
    with pytest.raises(ValueError, match='Unknown'):
        BridgeConfig.from_dict('x', {'command':['python'], 'runtime':'r', 'extra':True})


def test_process_is_lazy_and_correlates_hello_and_round_trip(tmp_path):
    bridge = ProcessBridge(config())
    assert bridge.process is None
    receipt = bridge.connect('session-1')
    assert receipt['runtime'] == 'synthetic-1'
    child = bridge.process
    value = bridge.request('session-1', 'request-1', 'echo', {'value':0.125})
    assert value == {'value':0.125,'session':'session-1','request':'request-1'}
    bridge.close()
    bridge.close()
    assert child.poll() is not None
    with pytest.raises(BridgeError):
        bridge.request('session-1', 'after-close', 'echo', {})


def test_acknowledged_duplicates_are_not_resent_and_conflicts_do_not_run(tmp_path):
    path = tmp_path/'effects.txt'
    bridge = ProcessBridge(config())
    try:
        bridge.connect('s')
        payload = {'path':str(path),'value':1}
        first = bridge.request('s','r','effect',payload)
        assert bridge.request('s','r','effect',payload) == first
        with pytest.raises(BridgeConflictError):
            bridge.request('s','r','effect',{'path':str(path),'value':2})
        assert path.read_text().count('effect') == 1
    finally:
        bridge.close()


def test_runtime_and_capability_mismatches_close_owned_child():
    for bad in (config(runtime='different'), config(capabilities=['unknown'])):
        bridge = ProcessBridge(bad)
        with pytest.raises(BridgeError):
            bridge.connect('s')
        assert bridge.process is None


def test_timeout_is_bounded_and_does_not_retry_or_kill_unrelated_process():
    unrelated = subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    bridge = ProcessBridge(config())
    try:
        bridge.connect('s')
        bridge.config = replace(bridge.config, timeout_seconds=0.2)
        started = time.monotonic()
        with pytest.raises(BridgeError, match='timed out'):
            bridge.request('s','slow','wait',{})
        assert time.monotonic()-started < 2
        assert bridge.process is None
        assert unrelated.poll() is None
        with pytest.raises(BridgeError):
            bridge.request('s','slow','wait',{})
    finally:
        bridge.close()
        unrelated.terminate()
        unrelated.wait(timeout=2)


def test_preparation_budget_does_not_extend_subsequent_decision_deadline():
    source = WORKER.replace("time.sleep(3)", "time.sleep(0.15)")
    bridge = ProcessBridge(config(command=[sys.executable, '-u', '-c', source]))
    try:
        bridge.connect('s')
        bridge.config = replace(bridge.config, timeout_seconds=0.05)
        bridge.request('s', 'prepare', 'wait', {}, timeout_seconds=0.5)
        assert bridge.config.timeout_seconds == 0.05
        with pytest.raises(BridgeError, match='timed out'):
            bridge.request('s', 'decision', 'wait', {})
    finally:
        bridge.close()


def test_blocked_stdin_write_is_included_in_request_timeout():
    source = r'''
import sys,json,time
first=json.loads(sys.stdin.readline())
print(json.dumps({'protocol':1,'session_id':first['session_id'],'request_id':first['request_id'],
                  'ok':True,'result':{'runtime':'synthetic-1','capabilities':['speech']}}),flush=True)
time.sleep(30)
'''
    bridge=ProcessBridge(config(command=[sys.executable,'-u','-c',source]))
    try:
        bridge.connect('s')
        bridge.config=replace(bridge.config,timeout_seconds=0.1)
        started=time.monotonic()
        with pytest.raises(BridgeError,match='timed out'):
            bridge.request('s','blocked','echo',{'value':'x'*60000})
        assert time.monotonic()-started < 2
        assert bridge.process is None
    finally:
        bridge.close()


def test_wrong_identity_oversize_and_worker_errors_close_child():
    raw = r'''
import sys,json
for line in sys.stdin:
    msg=json.loads(line)
    result={'runtime':'synthetic-1','capabilities':['speech']}
    response={'protocol':1,'session_id':msg['session_id'],'request_id':msg['request_id'],'ok':True,'result':result}
    if msg['operation']=='wrong': response['request_id']='other'
    if msg['operation']=='big': response['result']={'value':'x'*10000}
    print(json.dumps(response),flush=True)
'''
    for operation, command, size in [('wrong',raw,65536),('big',raw,1024),('fail',WORKER,65536)]:
        bridge=ProcessBridge(config(command=[sys.executable,'-u','-c',command],max_message_bytes=size))
        bridge.connect('s')
        with pytest.raises(BridgeError):
            bridge.request('s','r',operation,{})
        assert bridge.process is None


def test_requests_reject_nonfinite_or_oversized_payload_before_effects():
    bridge=ProcessBridge(config(max_message_bytes=1024))
    try:
        bridge.connect('s')
        for payload in ({'value':float('nan')},{'value':'x'*2000}):
            with pytest.raises((ValueError,BridgeError)):
                bridge.request('s','r','echo',payload)
    finally:
        bridge.close()


def test_worker_protocol_handles_duplicate_conflict_and_session_identity(tmp_path):
    path=tmp_path/'effects.txt'
    def message(s,r,value):
        return {'protocol':1,'session_id':s,'request_id':r,'operation':'effect',
                'payload':{'path':str(path),'value':value}}
    messages=[message('s','r',1),message('s','r',1),message('s','r',2),message('other','r',3)]
    proc=subprocess.run([sys.executable,'-u','-c',WORKER],input=''.join(json.dumps(m)+'\n'for m in messages),
                        text=True,capture_output=True,timeout=5,check=True)
    responses=[json.loads(line)for line in proc.stdout.splitlines()]
    assert responses[0] == responses[1]
    assert responses[2]['ok'] is False and 'identity' in responses[2]['error'].lower()
    assert responses[3]['ok'] is True
    assert path.read_text().count('effect') == 2


def test_worker_protocol_keeps_stdout_clean_and_closes_on_bad_input():
    source=WORKER.replace("return {'value':payload.get('value')", "print('sdk noise')\n        return {'value':payload.get('value')")
    msg={'protocol':1,'session_id':'s','request_id':'r','operation':'echo','payload':{}}
    proc=subprocess.run([sys.executable,'-u','-c',source],input=json.dumps(msg)+'\n',text=True,
                        capture_output=True,timeout=5,check=True)
    assert len(proc.stdout.splitlines()) == 1
    assert json.loads(proc.stdout)['ok'] is True
    assert 'sdk noise' in proc.stderr


def test_malformed_worker_message_never_calls_handler_and_closes(tmp_path):
    marker=tmp_path/'closed'
    source=WORKER.replace('        pass', '        open('+repr(str(marker))+",'w').write('closed')")
    proc=subprocess.run([sys.executable,'-u','-c',source],input='{"session_id":"s","session_id":"other"}\n',
                        text=True,capture_output=True,timeout=5,check=True)
    assert proc.stdout == ''
    assert marker.read_text() == 'closed'


def test_asset_manifest_requires_hashes_and_refuses_path_escape(tmp_path):
    model=tmp_path/'fixture.bin';model.write_bytes(b'inert test bytes')
    manifest=tmp_path/'assets.json'
    data={'source_url':'https://example.invalid/fixture','license':'fixture-only',
          'files':{'fixture.bin':hashlib.sha256(model.read_bytes()).hexdigest()}}
    manifest.write_text(json.dumps(data))
    assert verify_assets(manifest)[0] == tmp_path
    model.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):
        verify_assets(manifest)
    data['files']={'../outside': '0'*64};manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        verify_assets(manifest)
    data['files']={'fixture.bin':'not-a-digest'};manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        verify_assets(manifest)


def test_assets_reject_symbolic_links_even_with_matching_bytes(tmp_path):
    actual=tmp_path/'actual.bin';actual.write_bytes(b'fixture')
    link=tmp_path/'linked.bin'
    try:
        link.symlink_to(actual)
    except (OSError,NotImplementedError):
        pytest.skip('Symlink creation is not available for this test user.')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'source_url':'local fixture','license':'fixture',
                                   'files':{'linked.bin':hashlib.sha256(b'fixture').hexdigest()}}))
    with pytest.raises(ValueError,match='symbolic'):
        verify_assets(manifest)


def test_journal_deduplicates_identity_and_records_unknown_delivery(tmp_path):
    journal=SidecarJournal(tmp_path/'runtime.jsonl')
    first=journal.presentation_attempted('s','r',{'text':'unchanged'})
    assert journal.presentation_attempted('s','r',{'text':'unchanged'}) == first
    with pytest.raises(JournalError):
        journal.presentation_attempted('s','r',{'text':'different'})
    journal.presentation_failed('s','r','synthetic disconnect')
    rows=journal.read()
    assert [r['kind'] for r in rows] == ['presentation.attempted','presentation.failed']
    assert rows[-1]['payload']['delivery_status'] == 'unknown'
    assert not any('offer' in r['kind'] for r in rows)


def test_journal_failed_fsync_is_not_acknowledged_and_preserves_prior_bytes(tmp_path,monkeypatch):
    journal=SidecarJournal(tmp_path/'runtime.jsonl')
    journal.presentation_attempted('s','r',{})
    before=journal.path.read_bytes()
    def fail(fd): raise OSError('synthetic fsync failure')
    monkeypatch.setattr(os,'fsync',fail)
    with pytest.raises(JournalError):
        journal.presentation_delivered('s','r',{'delivered':True})
    assert journal.path.read_bytes() == before
    with pytest.raises(JournalError):
        journal.presentation_delivered('s','other',{'delivered':True})


def test_robot_worker_is_lazy_and_forwards_only_explicit_commands(monkeypatch):
    import types
    from submission_runtime.bridges import naoqi_legacy
    calls=[]
    class Proxy:
        def systemVersion(self): return 'synthetic-sdk'
        def getInstalledBehaviors(self): return ['approved']
        def runBehavior(self,name): calls.append(('gesture',name))
        def say(self,text): calls.append(('speech',text))
        def stiffnessInterpolation(self,*args): calls.append(('stiffness',args))
    def create(name,host,port):
        calls.append(('proxy',name))
        return Proxy()
    monkeypatch.setitem(sys.modules,'naoqi',types.SimpleNamespace(ALProxy=create))
    driver=naoqi_legacy.NaoqiDriver('configured.invalid',behaviors=['approved'])
    assert calls == []
    with pytest.raises(ValueError,match='Prepare'):
        driver.call('present',{'text':'unchanged'})
    monkeypatch.setattr(naoqi_legacy.sys,'version_info',(2,7))
    receipt=driver.call('hello',{})
    assert receipt['hardware_tested'] is False
    assert not any(x[0] in ['gesture','speech','stiffness']for x in calls)
    text='Approved \\pau=10\\ text'
    driver.call('present',{'text':text,'gesture':'approved'})
    assert calls[-2:] == [('gesture','approved'),('speech',text.encode('utf8'))]
    with pytest.raises(ValueError,match='configured'):
        driver.call('present',{'text':'unchanged','gesture':'unapproved'})
    with pytest.raises(ValueError,match='enabled'):
        driver.call('stiffness',{'value':0.5})
    driver.close()
    with pytest.raises(ValueError,match='Prepare'):
        driver.call('present',{'text':'unchanged'})
