"""Build an editable progress deck from verified reporting artifacts (no model runs)."""
from pathlib import Path
import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/presentation/unetpp_progress'
FIG = ROOT / 'outputs/unetpp_analysis'
OUT.mkdir(parents=True, exist_ok=True)
NAVY='173B48'; TEAL='007F7A'; MINT='8AD3C8'; CORAL='C45E48'; GRAY='526B75'; LIGHT='F3F7F7'; WHITE='FFFFFF'
prs=Presentation();prs.slide_width=Inches(13.3333);prs.slide_height=Inches(7.5)
plan=[]

def box(slide,x,y,w,h,fill):
    shape=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string(fill);shape.line.fill.background();return shape

def text(slide,content,x,y,w,h,size=22,color=NAVY,bold=False):
    shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=shape.text_frame;tf.word_wrap=True;tf.margin_left=0;tf.margin_right=0;tf.margin_top=0;tf.margin_bottom=0
    for i,line in enumerate(content.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Arial';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color);p.space_after=Pt(8)
    return shape

def slide(title,kicker,seconds,notes,dark=False,source='Local verified artifacts · UNet++ RD 002'):
    s=prs.slides.add_slide(prs.slide_layouts[6]);bg=s.background.fill;bg.solid();bg.fore_color.rgb=RGBColor.from_string(NAVY if dark else WHITE)
    fg=WHITE if dark else NAVY
    text(s,kicker.upper(),.55,.34,12,.35,12,MINT if dark else TEAL,True)
    text(s,title,.55,.95,12.15,1.05,31,fg,True)
    box(s,.55,7.03,12.2,.015,MINT if dark else 'D8E3E4')
    text(s,source,.55,7.15,11.5,.2,9,MINT if dark else GRAY)
    text(s,f'{len(prs.slides):02d}',12.2,7.12,.5,.25,10,MINT if dark else GRAY)
    s.notes_slide.notes_text_frame.text=f'TARGET: {seconds} seconds\n\n{notes}\n\nSOURCE: {source}'
    plan.append({'slide':len(prs.slides),'title':title,'seconds':seconds,'notes':notes,'source':source})
    return s

def picture(s,name,x,y,w,h):
    from PIL import Image
    path=FIG/f'{name}.png'
    with Image.open(path) as im: ratio=im.width/im.height
    fw,fh=w,min(h,w/ratio)
    if fh==h:fw=h*ratio
    return s.shapes.add_picture(str(path), Inches(x+(w-fw)/2), Inches(y+(h-fh)/2),width=Inches(fw),height=Inches(fh))

def line(s,head,body,y,x=.65,w=11.9):
    text(s,head,x,y,w,.45,22,TEAL,True);text(s,body,x,y+.5,w,.8,21)

def table(s,headers,rows,x,y,width,rowh=.58,fontsize=18):
    cols=len(headers); shape=s.shapes.add_table(len(rows)+1,cols,Inches(x),Inches(y), Inches(width), Inches(rowh*(len(rows)+1)))
    t=shape.table
    for i,row in enumerate([headers]+rows):
        for j,value in enumerate(row):
            cell=t.cell(i,j);cell.text=str(value);cell.fill.solid();cell.fill.fore_color.rgb=RGBColor.from_string(NAVY if i==0 else ('F0F6F5' if i%2 else WHITE));cell.margin_left=Inches(.12);cell.margin_top=Inches(.10)
            for p in cell.text_frame.paragraphs:
                p.font.name='Arial';p.font.size=Pt(fontsize);p.font.color.rgb=RGBColor.from_string(WHITE if i==0 else NAVY);p.font.bold=(i==0)
    return t

s=slide('Retinal Detachment Analysis\nfrom Ocular Ultrasound','CSCI 555 · Progress update',25,
'''Our project investigates retinal-detachment classification from ocular ultrasound videos. In the previous presentation we introduced the medical motivation, the ERDES dataset, and our plan to build a reproducible baseline. Today we can report a completed single-model benchmark, including a separate frozen test evaluation. We will explain what worked, what the early pilots taught us, and the limits of the findings.''',True)
text(s,'From a verified data pipeline to a completed UNet++ benchmark',.6,3.25,11.8,.9,27,WHITE)
text(s,'Bijay  ·  Naman  ·  Ritu\nSt. Francis Xavier University',.6,5.25,10,.9,20,MINT)

s=slide('A quick recap: our task and motivation','01 · Project recap',50,
'''Last time, we explained that ultrasound provides a video rather than one isolated image. We selected ERDES as a labeled research dataset and narrowed our first task to retinal detachment versus non-retinal detachment. Non-RD includes both normal clips and posterior vitreous detachment, or PVD. That distinction matters because a model must learn more than simply whether a membrane is visible. We also highlighted that the dataset contains far more non-RD than RD clips. Our original plan was to verify the data, establish a reproducible reference, and then investigate errors or a controlled improvement. This update follows that sequence. Macular-status classification is outside this completed experiment.''',source='Previous team presentation · ERDES v2, dataset/task definitions')
line(s,'Input','Ocular ultrasound video clips',2.15)
line(s,'First task','RD versus Non-RD; Non-RD includes Normal and PVD',3.55)
line(s,'Evaluation focus','Detect RD while tracking missed cases and false alarms',4.95)

s=slide('The planned baseline is now a measured result','02 · Progress since the last talk',45,
'''Our previous presentation ended with two technical gates: a verified data audit and a reproducible baseline. Both are now complete. We checked metadata and official split membership, inspected training videos, and tested input preprocessing. Next we ran two small CNN pilots to establish the training and evaluation pipeline. Finally, we trained the released UNet++ encoder-classifier equivalent for fifty epochs on the full official training set. We selected the checkpoint using validation only, froze it, and evaluated the official test set separately. The next contribution experiment has not yet been run, so we keep it separate from this completed reference.''')
for y,num,head,body in [(2.15,'01','Data audit','Official splits, labels and input checks completed'),(3.35,'02','Engineering pilots','Two five-epoch runs exposed the imbalance problem'),(4.55,'03','Full UNet++ baseline','50 epochs complete; epoch 48 selected using validation'),(5.75,'04','Frozen test evaluation','1,077 clips evaluated; predictions independently verified')]:
    text(s,num,.65,y,.65,.5,30,TEAL,True);text(s,head,1.6,y,3.7,.45,22,NAVY,True);text(s,body,5.1,y+.02,7.5,.7,20)

s=slide('The official split is strongly imbalanced','03 · Data and evaluation design',55,
'''The released binary task contains 5,381 clips: 4,879 non-RD and 502 RD. We retained the supplied assignments: 3,873 training clips, 431 validation clips, and 1,077 test clips. RD is roughly nine percent of each partition. Our integrity checks found no shared clip paths between partitions and labels agreed with metadata. However, this does not prove patient independence. The paper says identifiers were removed and the split is at clip level, so clips from one patient may occur in multiple partitions. We therefore describe this as performance on the official clip-level benchmark. The imbalance also explains why accuracy alone is insufficient: an always-non-RD predictor already reaches about ninety-one percent accuracy.''',source='Official split audit · ERDES v2, Data Records')
picture(s,'split_balance',.55,2.0,12.2,3.6)
text(s,'No clip overlap verified. Patient separation is not established.',.7,6.0,12,.6,21,CORAL,True)

s=slide('The pilot exposed a misleading accuracy score','04 · What we learned before full training',60,
'''Our first model was a compact custom 3D CNN, trained for only five epochs on 256 clips using sixteen sampled frames. It reported about ninety-one percent validation accuracy. The confusion counts revealed the problem: it predicted every validation clip as non-RD and missed all forty RD clips. We then tested positive-class weighting in the same small pilot. Average precision improved from about 0.091 to 0.145, but sensitivity at the fixed threshold remained zero. This was useful engineering evidence, not successful RD detection. The later full-data UNet++ run changes the model, training data amount, input duration and epoch budget together. We cannot attribute the resulting improvement to architecture alone. The pilots established that we must report sensitivity, specificity and ranking metrics alongside accuracy.''')
table(s,['Validation metric','Unweighted pilot','Weighted pilot'],[['Accuracy @ 0.5','90.72%','90.72%'],['RD sensitivity @ 0.5','0%','0%'],['Average precision','0.0907','0.1453'],['ROC-AUC','0.4491','0.6111']],.65,2.25,12,.64,20)
text(s,'Both pilots missed all 40 validation RD clips.',.7,5.95,12,.65,26,CORAL,True)

s=slide('One feasible reference: the UNet++ encoder classifier','05 · Model and fixed protocol',55,
'''We chose one feasible model rather than attempting all forty paper experiments. The released UNet++ model for this task is an encoder-based classifier, not a segmentation output. It has approximately 3.6 million parameters. We use the released pre-cropped clips, convert them to grayscale, square-pad, and interpolate the complete video to ninety-six frames of 128 by 128 pixels. The classifier uses 3D convolutional blocks, global average pooling and a binary head. Training follows the released Adam configuration, unweighted binary cross-entropy and fifty epochs. Our effective batch is forty-eight, implemented by accumulating twelve physical batches of four. We disclose differences from the reference, including mixed precision and decoding. Checkpoint selection remains highest validation accuracy with the earliest tie.''',source='Released ERDES code/config audit · configs/unetpp_rd_002_training.json')
for x,w,head,body in [( .65,3.6,'INPUT','1 × 96 × 128 × 128\nGrayscale video volume'),(4.85,3.6,'ENCODER','3D convolution blocks\n3.60M parameters'),(9.05,3.6,'OUTPUT','Global pooling + head\nRD score')]:
    box(s,x,2.45,w,1.65,LIGHT);text(s,head,x+.2,2.65,w-.4,.4,17,TEAL,True);text(s,body,x+.2,3.15,w-.4,.8,20)
text(s,'50 epochs  /  Adam 1.5 × 10⁻⁵  /  effective batch 48  /  seed 42',.7,4.6,12,.7,23,NAVY,True)
text(s,'Checkpoint: validation accuracy · Decision threshold: 0.5',.7,5.65,12,.7,23)

s=slide('A resumable workflow made free GPU compute workable','06 · Execution and reproducibility',55,
'''The full experiment exceeded our remaining Colab allocation, so we moved to Kaggle. To avoid spending GPU quota downloading and preparing videos, we prepared the training and validation inputs in two CPU jobs. Each output was saved, and the GPU jobs verified the cached tensor hashes before training. Training was resumed across three saved jobs, preserving model, optimizer, scheduler, scaler and random states. The allocation was T4 times two, but our implementation used one GPU. For test evaluation, we froze epoch forty-eight and threshold point-five before the evaluation workflow accessed test inputs. CPU preparation then saved the test cache, and a separate inference-only GPU job used the original training container. This makes the run traceable and avoids confusing resumed training with a new experiment.''')
line(s,'CPU preparation','Two saved shards → 4,304 verified train/validation inputs',2.05)
line(s,'GPU training','Saved jobs 01 → 02 → 03 resumed the same 50-epoch run',3.45)
line(s,'Separate test stage','Freeze epoch 48 → CPU test cache → inference-only GPU job',4.85)
text(s,'Reproducibility: configuration, hashes, predictions and checkpoints retained',.7,6.25,12,.45,17,GRAY)

s=slide('Learning curves show when RD detection emerged','07 · Training and validation',65,
'''The left panel shows training and validation binary cross-entropy over all fifty epochs. Both decrease substantially. The right panel explains why looking only at accuracy would have hidden early failure. Validation accuracy initially stays near ninety-one percent while sensitivity is zero. Average precision improves earlier, showing better score ordering before many RD scores cross the fixed threshold. Sensitivity starts rising later. The dotted vertical line is epoch forty-eight, selected by our pre-specified validation-accuracy rule. Its validation accuracy is 98.84 percent and sensitivity is 92.5 percent. Epoch forty-nine ties on accuracy and does not replace the earlier checkpoint. Epoch fifty is the final training state, but it is not the selected model. These are validation findings; the next slides show the separate official test evaluation.''')
picture(s,'learning_curves',.55,1.95,12.2,4.35)
text(s,'Epoch 48 selected using validation only; epoch 49 tied.',.7,6.42,12,.35,19,TEAL,True)

s=slide('Official test: 86 RD detected, 14 missed','08 · Frozen test performance',60,
'''This confusion matrix is the main test result. At the frozen threshold, the model correctly identified eighty-six RD clips and missed fourteen, out of one hundred RD clips. Among 977 non-RD clips, it correctly rejected 964 and raised thirteen false alarms. This gives 97.49 percent accuracy, 86 percent sensitivity and 98.67 percent specificity. The cell colors are normalized within each dataset class, while the large numbers show raw counts. This prevents the much larger negative class from visually hiding the RD errors. All 1,077 saved predictions were checked against the test manifest and all twelve exported artifact hashes were verified. These outcomes describe agreement with dataset labels. They do not establish clinical performance or performance on an independent hospital population.''')
picture(s,'test_confusion',.55,1.95,7.3,4.8)
for y,value,label in [(2.45,'97.49%','Accuracy'),(3.8,'86.00%','RD sensitivity'),(5.15,'98.67%','Specificity')]:text(s,value,8.6,y,4,.65,36,TEAL,True);text(s,label,8.65,y+.72,3.8,.4,20)

s=slide('Ranking is strong; the operating point remains fixed','09 · ROC and precision–recall',65,
'''The ROC curve summarizes sensitivity against the false-positive rate. Its area is approximately 0.987. Because RD is uncommon, we also show precision against recall. Average precision is approximately 0.927, compared with an RD prevalence of about 0.093. Here average precision uses the standard non-interpolated statistic, rather than being relabeled as trapezoidal PR area. The marked point on both curves is threshold point-five, which we fixed before test inference. We are not selecting a more favorable test threshold. These plots describe ranking across this test partition; the scores have not been calibrated as clinical risk probabilities. A strong ROC curve also does not erase the fourteen missed RD clips at the reported operating point. We therefore show curves and confusion counts together.''')
picture(s,'test_curves',.55,1.95,12.2,4.45)
text(s,'ROC-AUC 0.9869  ·  Average precision 0.9268  ·  Threshold 0.5 unchanged',.7,6.4,12,.4,20,TEAL,True)

s=slide('Our test result is close to the published UNet++ row','10 · Test-to-test benchmark comparison',60,
'''We compare our official test result to the correct row: UNet++ for non-RD versus RD in Table eight. We do not compare our selected validation score to the paper's test score. Our accuracy is 97.49 percent, against the reported 97.6 percent. Sensitivity is 86 percent, against 87.9 percent; specificity is 98.67 percent, against 98.6 percent. Precision and F1 are also shown. This establishes one feasible adapted benchmark close to the published result. It does not establish exact numerical replication or statistical equivalence. Important differences remain: the paper text says AdamW while the released configuration uses Adam, and our implementation uses a different decoder, gradient accumulation and mixed precision. The paper's sensitivity denominator or aggregation is also unresolved, so we retain its reported value without inventing an explanation.''',source='Ozkut et al., ERDES v2, Table 8 · https://arxiv.org/html/2508.04735v2')
table(s,['Metric','Our official test','Paper test'],[['Sensitivity','86.00%','87.9%'],['Specificity','98.67%','98.6%'],['Precision','86.87%','87.1%'],['F1','86.43%','87.5%'],['Accuracy','97.49%','97.6%']],.65,2.05,12,.59,20)
text(s,'One adapted benchmark completed; exact numerical replication is not claimed.',.7,6.25,12,.55,20,TEAL,True)

s=slide('The remaining errors and limits shape our next study','11 · Interpretation and limitations',70,
'''The result is encouraging as a course benchmark, but its limitations determine what we can claim. First, sensitivity depends on only one hundred RD test clips. A change of one detected RD clip moves sensitivity by one percentage point, and fourteen were missed. Second, the official split is clip-level, so patient overlap may inflate apparent generalization. Third, we have one seed, one official split and no external dataset. Fourth, execution differs numerically from the reference, and a single mixed-precision optimizer update was skipped in epoch thirty-five and recorded. We do not interpret this result as clinical validation or a robust improvement over the paper. Finally, the test set is now exposed. Future experiment choices must use training and validation data, and any later comparison on the same official test must be disclosed as reuse. The useful next contribution is a pre-specified controlled study of errors or imbalance, not repeated test-driven tuning.''')
line(s,'Missed RD remains the key error','14 / 100 RD clips missed at the frozen operating point',2.05)
line(s,'Benchmark scope is limited','Clip-level split, possible patient overlap, one seed, no external validation',3.45)
line(s,'Protocol differences are disclosed','Adam/code vs. AdamW/text; decoder, accumulation and mixed precision',4.85)
text(s,'Educational research · No clinical validation claimed',.7,6.3,12,.45,19,CORAL,True)

s=slide('A completed reference, then one controlled extension','12 · Next milestone',55,
'''We now have the technical reference that was missing at our previous progress update: a verified pipeline, full fifty-epoch training, a frozen checkpoint, official-test predictions and a documented comparison with the paper. The immediate deliverable is this progress report with the results notebook and reusable figures. For the next experiment, we propose a single controlled question about imbalance handling, such as whether training-only class weighting improves validation sensitivity while tracking specificity. That proposal needs a written protocol before execution and should keep the model, inputs, splits and compute budget comparable. It is not yet a completed contribution or a promised improvement. We will preserve the current reference and disclose that the official test has already been evaluated. This gives the project a clear baseline and a defensible path to its own analysis.''',True)
text(s,'COMPLETED',.7,2.35,5.8,.5,18,MINT,True);text(s,'Verified pipeline\n50-epoch UNet++ baseline\nFrozen official-test evaluation',.7,3.05,5.8,2.5,27,WHITE)
text(s,'PROPOSED NEXT',7.0,2.35,5.6,.5,18,MINT,True);text(s,'Pre-specify one comparison\nChoose using train/validation\nPreserve the reference result',7.,3.05,5.65,2.5,25,WHITE)
text(s,'Questions and discussion',.7,6.2,11,.6,27,MINT,True)

# Backup slides are outside the 12-minute speaking budget.
s=slide('Protocol and reproducibility details','Appendix A · For questions',0,
'''Backup only. The input uses full-video temporal/spatial interpolation, rather than sixteen evenly sampled frames. We use the released encoder-classifier equivalent, unweighted BCE, Adam and fixed fifty-epoch budget. Data and checkpoint identities are retained. Validation was used for checkpoint selection; no optimizer or threshold search was used during test evaluation.''')
table(s,['Setting','Recorded value'],[['Input / model','1×96×128×128 / 3,603,553 parameters'],['Optimizer / learning rate','Adam / 1.5e-5; no weight decay'],['Batch / precision','4 physical × 12 accumulation / 16-mixed'],['Selection / threshold','Max validation accuracy, earliest tie / 0.5'],['Training / evaluation','50 epochs / frozen epoch 48'],['Verification','70 training + 12 evaluation artifact hashes']],.65,2.05,12,.59,18)

s=slide('Sources and reusable evidence','Appendix B · References',0,
'''Backup only. The original team presentation supplies the recap. The ERDES v2 article and released code/config audit provide the published comparison and protocol. The local verified archives provide our measurements. The results notebook exports every figure and table without retraining. See the repository report for complete SHA256 identities and limitations.''',source='Sources and artifacts; links are included in notes and the local speaker guide')
line(s,'Published reference','Ozkut et al. · ERDES v2 · arxiv.org/abs/2508.04735',2.05)
line(s,'Project record','docs/UNETPP_RESULTS.md · docs/EXPERIMENT_LOG.md',3.45)
line(s,'Reproducible visuals','notebooks/unetpp_results_analysis.ipynb → outputs/unetpp_analysis/',4.85)

assert sum(p['seconds'] for p in plan)==720
prs.save(OUT/'unetpp_progress_12min.pptx')
(OUT/'slide_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
guide=['# UNet++ progress presentation — 12-minute speaker guide','',
'13 main slides; 12 minutes of speaking. Questions afterward. Two appendix slides have no allocated talk time.',
'Team names follow the previous deck: Bijay, Naman and Ritu. Rehearse handoffs and divide speaking roles together.',
'Figures are derived from saved artifacts, without retraining or test threshold tuning.','']
elapsed=0
for p in plan:
    start=elapsed;elapsed+=p['seconds']
    guide += [f"## {p['slide']:02d}. {p['title']}",f"Time: {start//60}:{start%60:02d}–{elapsed//60}:{elapsed%60:02d}" if p['seconds'] else 'Backup slide — outside speaking budget','',p['notes'],'',f"Source: {p['source']}",'']
(ROOT/'docs/UNETPP_PRESENTATION_GUIDE.md').write_text('\n'.join(guide))
print(f'Saved {len(prs.slides)} slides; timed main section = {elapsed} seconds. PPTX: {OUT}')
