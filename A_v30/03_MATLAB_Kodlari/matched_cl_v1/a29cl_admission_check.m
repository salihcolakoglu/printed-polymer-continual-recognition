function receipt=a29cl_admission_check(outDir)
% Read actual source only for metadata/scaler admission, no model call/fit.
A=fileparts(fileparts(fileparts(mfilename('fullpath'))));
D=a29cl_data('prepare',fullfile(A,'02_Veriler','A_source_average_records_v1.csv'), ...
 fullfile(A,'02_Protokol','A29_Matched_CL_Input_Links_v1.json'));
receipt=D.admission;receipt.source_training=0;receipt.source_predictions=0;
receipt.meaning='Actual CSV admission only. No learned model, inference, new scaler selection or source experiment.';
assert(~D.isfixture);
for f=1:5
 for stream={'TaskIL','ClassIL'}
  for stage=1:3
   b=D.current(f,stream{1},stage);
   assert(all(b.outer_fold~=f)&&all(ismember(b.y,b.current_labels)));
  end
 end
end
receipt.current_loader_membership_checks=30;
path=fullfile(outDir,'admission.json');assert(~isfile(path));text=jsonencode(receipt);b=unicode2native(text,'UTF-8');assert(numel(b)<65536);
f=fopen(path,'wb');assert(f>=0);c=onCleanup(@()fclose(f));fwrite(f,b,'uint8');clear c
end
