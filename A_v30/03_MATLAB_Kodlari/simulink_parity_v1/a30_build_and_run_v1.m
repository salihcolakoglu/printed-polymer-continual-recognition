function a30_build_and_run_v1(configPath)
% Bounded launcher verifies exact pins before invoking this driver.
cfg=jsondecode(fileread(configPath));out=cfg.out_dir;
assert(isfolder(out)); assert(~isfile(fullfile(out,'run_complete.json')));
cd(out); Simulink.fileGenControl('set','CacheFolder',fullfile(out,'cache'), ...
 'CodeGenFolder',fullfile(out,'codegen'),'createDir',true);
bank=jsondecode(fileread(cfg.bank_path));
theta=double(bank.theta);meta=double(bank.meta);mu=double(bank.mean);sd=double(bank.sd);
refs=double(bank.refs);books=double(bank.books);
assert(isequal(size(theta),[522,52]) && isequal(size(meta),[522,5]));
assert(isequal(size(refs),[252,3]) && isequal(size(books),[8,8]));
mu=reshape(mu,1,2);sd=reshape(sd,1,2);
% JSON null is NaN; no source query or recorded output is embedded in model.
name=sprintf('a30_inference_f%d_%s',cfg.fold,cfg.mode);
assert(~bdIsLoaded(name));new_system(name);
cleanup=onCleanup(@()close_owned(name)); %#ok<NASGU>
set_param(name,'SolverType','Fixed-step','Solver','FixedStepDiscrete', ...
 'FixedStep','1','StartTime','0','StopTime','1','ReturnWorkspaceOutputs','on', ...
 'SaveTime','on','TimeSaveName','tout','SignalLogging','off');
ws=get_param(name,'ModelWorkspace');
names={'theta','meta','mu','sd','refs','books'};
values={theta,meta,mu,sd,refs,books};
assignin(ws,'queries',timeseries(zeros(2,7),[0;1]));
add_block('simulink/Sources/From Workspace',[name,'/Raw records and eligibility'], ...
 'VariableName','queries','SampleTime','1','Interpolate','off', ...
 'OutputAfterFinalValue','Holding final value','Position',[25,70,195,110]);
add_block('simulink/Signal Attributes/Signal Specification',[name,'/Seven channels'], ...
 'Dimensions','7','OutDataTypeStr','double','Position',[225,70,325,110]);
add_line(name,'Raw records and eligibility/1','Seven channels/1');
block=[name,'/Normalize and infer'];
add_block('simulink/User-Defined Functions/MATLAB Function',block,'Position',[470,65,700,360]);
chart=find(sfroot,'-isa','Stateflow.EMChart','Path',block);
source=fileread(fullfile(cfg.code_dir,'a30_inference_chart_v1.m'));
chart.Script=strrep(source,'function y = a30_inference_chart_v1(u,theta,meta,mu,sd,refs,books)', ...
 'function y = fcn(u,theta,meta,mu,sd,refs,books)');
add_line(name,'Seven channels/1','Normalize and infer/1');
for j=1:6
 assignin(ws,names{j},values{j});
 add_block('simulink/Sources/Constant',[name,'/',names{j}], ...
  'Value',names{j},'Position',[245,150+j*50,355,180+j*50]);
 add_line(name,[names{j},'/1'],['Normalize and infer/',num2str(j+1)]);
end
add_block('simulink/Sinks/To Workspace',[name,'/Full numeric trace'], ...
 'VariableName','trace','SaveFormat','Array','MaxDataPoints','inf', ...
 'Decimation','1','SampleTime','1','Position',[770,165,905,205]);
add_line(name,'Normalize and infer/1','Full numeric trace/1');
set_param(name,'Description', ...
 'A30 saved A29 stage-bank inference. Raw modulus MPa and ShoreD -> frozen first-stage TRAIN normalizer -> frozen saved state -> eligible scores and exact tie rule. Tick is query index. No training, physical dynamics, truth or expected outputs. MIT authored implementation; bank terms documented separately.');
modelPath=fullfile(out,[name,'.slx']);save_system(name,modelPath);
close_system(name,0);
% Mandatory cold load: inference uses the actual saved self-contained model.
load_system(modelPath);ws=get_param(name,'ModelWorkspace');
for j=1:6,assert(isequaln(evalin(ws,names{j}),values{j}));end
chart=find(sfroot,'-isa','Stateflow.EMChart','Path',block);
assert(isequal(chart.Script,strrep(source,'function y = a30_inference_chart_v1(u,theta,meta,mu,sd,refs,books)', ...
 'function y = fcn(u,theta,meta,mu,sd,refs,books)')));
queries=readmatrix(cfg.inputs_path);assert(size(queries,2)==7);
assert(size(queries,1)==cfg.expected_rows);
assignin(ws,'queries',timeseries(queries,(0:size(queries,1)-1)'));
set_param(name,'StopTime',num2str(size(queries,1)-1));
tic;simout=sim(name);elapsed=toc;
trace_raw=simout.get('trace');time=simout.get('tout');
raw_trace_size=size(trace_raw);N=size(queries,1);
if isequal(size(trace_raw),[N,35])
 trace=trace_raw;axes='samples_by_channels';
elseif isequal(size(trace_raw),[35,1,N])
 trace=permute(trace_raw,[3,1,2]);axes='channels_singleton_samples_permute_312';
elseif isequal(size(trace_raw),[1,35,N])
 trace=permute(trace_raw,[3,2,1]);axes='singleton_channels_samples_permute_321';
else,error('A30SLX:TraceAxes','Unexpected ToWorkspace trace axes.');end
assert(isequal(size(trace),[size(queries,1),35]));
assert(isequal(time(:),(0:size(queries,1)-1)'));
writematrix(trace,fullfile(out,'actual_trace.csv'));
write_json(fullfile(out,'actual_trace.json'),struct('time',time(:),'values',trace));
save(fullfile(out,'actual_trace.mat'),'trace','trace_raw','time','-v7');
semantic=struct('schema','A30_SAVED_SLX_READBACK_V1','mode',cfg.mode,'fold',cfg.fold, ...
 'cold_reopened',true,'model_path',modelPath,'theta',evalin(ws,'theta'), ...
 'meta',evalin(ws,'meta'),'mean',evalin(ws,'mu'),'sd',evalin(ws,'sd'), ...
 'refs',evalin(ws,'refs'),'books',evalin(ws,'books'),'chart_script',chart.Script, ...
 'blocks',{find_system(name,'SearchDepth',1,'Type','Block')}, ...
 'lines',{get_param(find_system(name,'FindAll','on','SearchDepth',1,'Type','line'),'Name')}, ...
 'rows',size(trace,1),'columns',35,'raw_trace_size',raw_trace_size,'trace_axes',axes,'simulation_seconds',elapsed, ...
 'MATLAB_version',version,'Simulink_version',ver('Simulink'));
write_json(fullfile(out,'saved_model_semantic_readback.json'),semantic);
write_json(fullfile(out,'run_complete.json'),struct('schema','A30_RUN_COMPLETE_V1', ...
 'mode',cfg.mode,'fold',cfg.fold,'rows',size(trace,1),'sim_seconds',elapsed, ...
 'cold_reopened',true,'training',false,'physical_validation',false));
end
function close_owned(name)
if bdIsLoaded(name),close_system(name,0);end
end
function write_json(p,s)
f=fopen(p,'w','n','UTF-8');assert(f>=0);c=onCleanup(@()fclose(f)); %#ok<NASGU>
fprintf(f,'%s\n',jsonencode(s));
end
