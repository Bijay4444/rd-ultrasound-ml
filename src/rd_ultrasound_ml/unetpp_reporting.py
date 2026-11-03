"""Read-only reporting for the frozen UNet++ experiment; no training or tuning."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score)

TEAL = '#007F7A'
NAVY = '#173B48'
CORAL = '#C45E48'
GRAY = '#60737C'
PAPER = 'https://arxiv.org/html/2508.04735v2'

def configure_style():
    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 220,
        'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 15,
        'axes.titleweight': 'bold', 'axes.labelcolor': NAVY, 'text.color': NAVY,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#CCD6D8', 'grid.color': '#E1E7E8',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})

def load_verified_run(root):
    root = Path(root)
    archives = {
        'training': root/'outputs/kaggle_results/session_03/unetpp_rd_002_amp_results.zip',
        'test': root/'outputs/kaggle_results/test_evaluation/unetpp_rd_002_test_results.zip'}
    payloads, audit = {}, {}
    for kind, path in archives.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}. Retrieve the saved results ZIP; see docs/UNETPP_RESULTS.md.')
        with zipfile.ZipFile(path) as z:
            hash_names = [n for n in z.namelist() if n.endswith('/artifact_sha256.json')]
            if len(hash_names) != 1:
                raise ValueError('Expected one artifact hash manifest')
            prefix = hash_names[0].rsplit('/', 1)[0] + '/'
            hashes = json.loads(z.read(hash_names[0]))
            for name, digest in hashes.items():
                if hashlib.sha256(z.read(prefix+name)).hexdigest() != digest:
                    raise ValueError(f'Artifact hash mismatch: {kind}/{name}')
            payloads[kind] = {n[len(prefix):]: z.read(n) for n in z.namelist()
                if n.startswith(prefix) and n.endswith(('.json', '.csv'))}
        audit[kind] = {'archive_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'artifact_hashes_verified': len(hashes)}
    from io import BytesIO
    training, test = payloads['training'], payloads['test']
    summary = json.loads(training['summary.json'])
    history = pd.read_csv(BytesIO(training['history.csv']))
    result = json.loads(test['test_metrics.json'])
    predictions = pd.read_csv(BytesIO(test['test_predictions.csv']))
    manifest = pd.read_csv(BytesIO(test['test_manifest.csv']))
    freeze = json.loads((root/'configs/unetpp_rd_002_evaluation.json').read_text())
    archived_freeze = json.loads(test['evaluation_freeze.json'])
    assert freeze == archived_freeze, 'Frozen evaluation config changed'
    assert summary['status'] == 'completed' and summary['epochs_completed'] == 50
    assert history.epoch.tolist() == list(range(1, 51))
    assert int(history.loc[history.accuracy.idxmax(), 'epoch']) == summary['best_epoch'] == result['selected_epoch'] == 48
    assert result['checkpoint_sha256'] == freeze['checkpoint_sha256']
    assert result['threshold_tuned'] is False and result['training_performed'] is False
    assert len(predictions) == len(manifest) == result['test_clips'] == 1077
    for col in ['clip_id', 'file_path', 'label']:
        assert predictions[col].tolist() == manifest[col].tolist(), col
    p = predictions.rd_probability.to_numpy()
    y = predictions.label.to_numpy()
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert predictions.clip_id.is_unique and (y == 1).sum() == 100 and (y == 0).sum() == 977
    hard = (p >= freeze['threshold']).astype(int)
    np.testing.assert_array_equal(hard, predictions.predicted_label)
    tn, fp, fn, tp = confusion_matrix(y, hard, labels=[0, 1]).ravel()
    metrics = {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'sensitivity': tp/(tp+fn), 'specificity': tn/(tn+fp),
        'precision': tp/(tp+fp), 'F1': 2*tp/(2*tp+fp+fn),
        'accuracy': (tp+tn)/len(y), 'balanced_accuracy': (tp/(tp+fn)+tn/(tn+fp))/2,
        'average_precision': average_precision_score(y,p), 'ROC_AUC': roc_auc_score(y,p)}
    for key, value in metrics.items():
        assert np.isclose(value, result['metrics'][key], rtol=0, atol=1e-12), key
    assert json.loads(test['runtime.json']) == json.loads(training['runtime.json'])
    audit.update({'prediction_rows_verified': len(y), 'selected_epoch': 48,
        'threshold': freeze['threshold'], 'metrics_recomputed': True, 'no_training_or_tuning': True})
    return {'summary': summary, 'history': history, 'result': result,
        'predictions': predictions, 'freeze': freeze, 'metrics': metrics, 'audit': audit}

def make_tables(run):
    m = run['metrics']; v = run['summary']['best_validation_metrics']
    splits = pd.DataFrame({'Partition':['Train','Validation','Test'],
        'Non-RD':[3511,391,977], 'RD':[362,40,100]}).set_index('Partition')
    splits['Total'] = splits['Non-RD'] + splits['RD']
    splits['RD fraction'] = splits.RD/splits.Total
    names=['sensitivity','specificity','precision','F1','accuracy','balanced_accuracy','average_precision','ROC_AUC']
    metrics = pd.DataFrame({'Selected validation':[v[n] for n in names],
        'Official test':[m[n] for n in names]}, index=names)
    reference = run['freeze']['published_reference']
    comparison=pd.DataFrame({'Our test':[m[n] for n in names[:5]],
        'Paper test':[reference[n] for n in names[:5]]},index=names[:5])
    comparison['Difference (percentage points)']=100*(comparison['Our test']-comparison['Paper test'])
    confusion=pd.DataFrame([[m['TN'],m['FP']],[m['FN'],m['TP']]],
        index=['Dataset Non-RD','Dataset RD'],columns=['Predicted Non-RD','Predicted RD'])
    pilot=pd.DataFrame({'Experiment':['Unweighted pilot','Weighted pilot'],
        'Training clips':[256,256], 'Frames':[16,16], 'Epochs':[5,5],
        'Validation AP':[0.09068340359421731,0.1453199778],
        'Validation ROC-AUC':[0.449105,0.611061], 'RD sensitivity @0.5':[0.,0.]})
    # Pilot values are transcribed from the executed notebook, not directly comparable test runs.
    return {'split_summary':splits, 'metric_summary':metrics, 'paper_comparison':comparison,
            'test_confusion':confusion, 'pilot_summary':pilot}

def wilson_interval(successes, total, z=1.959963984540054):
    p=successes/total; denom=1+z*z/total
    centre=(p+z*z/(2*total))/denom
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return centre-half, centre+half

def create_figures(run, output_dir):
    configure_style(); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    figures={}; tables=make_tables(run); h=run['history']; m=run['metrics']
    p=run['predictions'].rd_probability.to_numpy(); y=run['predictions'].label.to_numpy()
    def save(name, fig):
        fig.savefig(out/f'{name}.png',bbox_inches='tight',facecolor='white')
        fig.savefig(out/f'{name}.pdf',bbox_inches='tight',facecolor='white')
        figures[name]=fig
    fig,ax=plt.subplots(figsize=(9,3.8)); s=tables['split_summary']
    x=np.arange(3); ax.barh(x,s['Non-RD'],color=TEAL,label='Non-RD');ax.barh(x,s.RD,left=s['Non-RD'],color=CORAL,label='RD')
    for i, row in enumerate(s.itertuples()):
        ax.text(row.Total+45,i,f'{row.Total:,} clips | {row.RD_fraction:.1%} RD' if hasattr(row,'RD_fraction') else f'{row.Total:,} clips | {row.RD/row.Total:.1%} RD',va='center',fontsize=10)
    ax.set(yticks=x,yticklabels=s.index,xlabel='Number of clips',xlim=(0,5100),title='Official splits preserve approximately 9.3% RD');ax.invert_yaxis();ax.legend(loc='lower right');save('split_balance',fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4));ax=axes[0]
    ax.plot(h.epoch,h.train_loss,color=TEAL,label='Training BCE');ax.plot(h.epoch,h.validation_loss,color=CORAL,label='Validation BCE')
    ax.set(xlabel='Epoch',ylabel='Unweighted BCE',title='Loss decreases across 50 epochs');ax.legend()
    ax=axes[1];ax.plot(h.epoch,h.sensitivity,color=CORAL,label='Validation sensitivity');ax.plot(h.epoch,h.average_precision,color=TEAL,label='Validation AP');ax.plot(h.epoch,h.accuracy,color=NAVY,ls='--',label='Validation accuracy')
    ax.set(xlabel='Epoch',ylabel='Score',ylim=(-.03,1.06),title='Accuracy alone hides early missed RD');ax.legend(fontsize=9,loc='lower right')
    for ax in axes:ax.axvline(48,color=GRAY,ls=':',label='Selected epoch 48');ax.grid(alpha=.5)
    fig.tight_layout();save('learning_curves',fig)
    fig,ax=plt.subplots(figsize=(6,4.5));cm=np.array([[m['TN'],m['FP']],[m['FN'],m['TP']]])
    # Color by row fraction so class imbalance does not hide RD errors; labels are raw counts.
    frac=cm/cm.sum(axis=1,keepdims=True);ax.imshow(frac,cmap='Blues',vmin=0,vmax=1)
    for (i,j), count in np.ndenumerate(cm):ax.text(j,i,f'{count}\n({frac[i,j]:.1%} of row)',ha='center',va='center',fontsize=16,color='white' if frac[i,j]>.6 else NAVY)
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Non-RD','RD'],yticklabels=['Non-RD (n=977)','RD (n=100)'],xlabel='Model prediction at threshold 0.5',ylabel='Dataset label',title='Official test: 14 missed RD, 13 false alarms');save('test_confusion',fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.1));fpr,tpr,_=roc_curve(y,p);prec,rec,_=precision_recall_curve(y,p)
    axes[0].plot(fpr,tpr,color=TEAL,lw=2,label=f'ROC-AUC = {m["ROC_AUC"]:.3f}');axes[0].plot([0,1],[0,1],color=GRAY,ls='--');axes[0].scatter([m['FP']/977],[m['sensitivity']],color=CORAL,zorder=5,label='Fixed threshold 0.5')
    axes[0].set(xlabel='False-positive rate',ylabel='Sensitivity',title='Official-test ROC',xlim=(-.02,1.02),ylim=(-.02,1.02));axes[0].legend(loc='lower right')
    axes[1].plot(rec,prec,color=TEAL,lw=2,label=f'Average precision = {m["average_precision"]:.3f}');axes[1].axhline(y.mean(),color=GRAY,ls='--',label=f'RD prevalence = {y.mean():.3f}');axes[1].scatter([m['sensitivity']],[m['precision']],color=CORAL,zorder=5,label='Fixed threshold 0.5')
    axes[1].set(xlabel='Recall / sensitivity',ylabel='Precision',title='Official-test precision–recall',xlim=(-.02,1.02),ylim=(-.02,1.02));axes[1].legend(loc='lower left',fontsize=9)
    for ax in axes:ax.grid(alpha=.5)
    fig.tight_layout();save('test_curves',fig)
    fig,ax=plt.subplots(figsize=(9,4.2));c=tables['paper_comparison'];x=np.arange(len(c));width=.34
    bars=ax.bar(x-width/2,100*c['Our test'],width,color=TEAL,label='Our official test');other=ax.bar(x+width/2,100*c['Paper test'],width,color='#9EAFB4',label='Paper Table 8')
    ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9);ax.bar_label(other,fmt='%.1f',padding=3,fontsize=9)
    ax.set(xticks=x,xticklabels=['Sensitivity','Specificity','Precision','F1','Accuracy'],ylabel='Percent',ylim=(0,112),title='One adapted UNet++ benchmark is close to the published row');ax.legend(loc='lower left');ax.grid(axis='y',alpha=.4);save('paper_comparison',fig)
    fig,ax=plt.subplots(figsize=(9,3.7));bins=np.linspace(0,1,26)
    for label,color,name in [(0,TEAL,'Dataset Non-RD'),(1,CORAL,'Dataset RD')]:ax.hist(p[y==label],bins=bins,weights=np.ones((y==label).sum())/(y==label).sum(),alpha=.65,color=color,label=name)
    ax.axvline(.5,color=NAVY,ls='--',label='Fixed threshold 0.5');ax.set(xlabel='Model RD score (calibration not established)',ylabel='Fraction within dataset class',title='Score distributions describe separation, not calibrated risk');ax.legend();save('score_distribution',fig)
    fig,ax=plt.subplots(figsize=(8,3.6));labels=['Sensitivity\n86 / 100 RD','Specificity\n964 / 977 Non-RD','Accuracy\n1050 / 1077 clips'];vals=[m['sensitivity'],m['specificity'],m['accuracy']];intervals=[wilson_interval(86,100),wilson_interval(964,977),wilson_interval(1050,1077)]
    for i,(v,(lo,hi)) in enumerate(zip(vals,intervals)):ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=TEAL,capsize=5,ms=8);ax.text(1.02,i,f'{v:.1%}  [{lo:.1%}, {hi:.1%}]',transform=ax.get_yaxis_transform(),va='center',fontsize=10)
    ax.set(yticks=range(3),yticklabels=labels,xlim=(.72,1.01),ylim=(2.6,-.6),xlabel='Score with illustrative 95% Wilson interval',title='Only 100 RD test clips limit sensitivity precision');ax.grid(axis='x',alpha=.4);save('metric_intervals',fig)
    for name,table in tables.items():table.to_csv(out/f'{name}.csv')
    (out/'report_audit.json').write_text(json.dumps(run['audit'],indent=2)+'\n')
    for fig in figures.values():
        plt.close(fig)  # Notebook cells display only their requested figure.
    return figures


def display_table(frame, formatters=None):
    """Render readable tables without pandas' optional Jinja2 Styler dependency."""
    from IPython.display import HTML, display
    css = """<style>.rd-results {border-collapse:collapse;font-family:Arial,sans-serif;font-size:14px;}
    .rd-results th {background:#173B48;color:white;padding:10px;text-align:left;}
    .rd-results td {padding:10px;border-bottom:1px solid #D8E3E4;text-align:right;}
    .rd-results tr:nth-child(even) td {background:#F0F6F5;}</style>"""
    display(HTML(css + frame.to_html(classes='rd-results', border=0,
        formatters=formatters, float_format=lambda x: f'{x:.4f}')))
