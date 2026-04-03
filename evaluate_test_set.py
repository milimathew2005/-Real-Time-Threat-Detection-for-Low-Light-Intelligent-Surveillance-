import sys
import glob
import os
os.environ['WERKZEUG_RUN_MAIN'] = 'true'
import app

def run_evaluation():
    print('[*] Loading models...')
    app.load_yolo()
    app.load_lstm()
    test_files = glob.glob('test/*.mp4')
    if not test_files:
        print('No test files found in test/ directory.')
        return
    print(f'[*] Found {len(test_files)} test videos. Starting inference...\n')
    correct = 0
    results = []
    for v in test_files:
        filename = os.path.basename(v)
        is_anomaly = 'Normal' not in filename
        job_id = 'test_' + filename
        app.analysis_jobs[job_id] = {'status': 'processing', 'progress': 0, 'result': None, 'error': None}
        try:
            app.analyse_video(job_id, v)
            res = app.analysis_jobs[job_id].get('result')
            if not res:
                print(f'[!] Error analyzing {filename}')
                continue
            yolo_weaps = res.get('weapon_detections', 0)
            lstm_lbl = res.get('lstm_summary', {}).get('label', 'Normal')
            system_flagged = yolo_weaps > 0 or lstm_lbl == 'Anomaly'
            msg = ''
            if is_anomaly and system_flagged:
                correct += 1
                msg = '✅ TRUE POSITIVE'
            elif not is_anomaly and (not system_flagged):
                correct += 1
                msg = '✅ TRUE NEGATIVE'
            elif is_anomaly and (not system_flagged):
                msg = '❌ FALSE NEGATIVE (Missed Threat)'
            elif not is_anomaly and system_flagged:
                msg = '❌ FALSE POSITIVE (False Alarm)'
            print(f'[{filename}] -> Expected Anomaly: {is_anomaly} | System Flagged: {system_flagged}')
            print(f'   ↳ LSTM: {lstm_lbl} | Max Weapons: {yolo_weaps} => {msg}\n')
        except Exception as e:
            print(f'[!] Exception on {v}: {e}')
    acc = correct / len(test_files) * 100
    print('-' * 50)
    print(f'🔥 FINAL PIPELINE ACCURACY: {correct}/{len(test_files)} ({acc:.1f}%)')
    print('-' * 50)
if __name__ == '__main__':
    run_evaluation()