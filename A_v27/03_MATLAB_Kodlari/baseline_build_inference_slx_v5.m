function modelPath = baseline_build_inference_slx_v5(versionRoot)
% Author-readable builder. It creates a real SLX only when MATLAB runs it.
% Requires MATLAB+Simulink; no Deep Learning Toolbox training API.
modelName='baseline_A_inference_v5';
modelPath=fullfile(versionRoot,'04_Simulink_Modelleri',[modelName,'.slx']);
assert(~isfile(modelPath),'Baseline:ExistingModel','Refusing to overwrite an existing SLX.');
assert(~bdIsLoaded(modelName),'Baseline:LoadedModel','A same-name model is already loaded.');
new_system(modelName); cleanup=onCleanup(@()close_system(modelName,0)); %#ok<NASGU>
set_param(modelName,'SolverType','Fixed-step','Solver','FixedStepDiscrete', ...
 'FixedStep','1','StartTime','0','StopTime','23','SaveTime','on', ...
 'TimeSaveName','tout','ReturnWorkspaceOutputs','on');
ws=get_param(modelName,'ModelWorkspace');
assignin(ws,'baseline_inputs',timeseries(zeros(2,4),[0;1]));
assignin(ws,'baseline_p',zeros(32,1)); assignin(ws,'baseline_mu',[0,0]);
assignin(ws,'baseline_sd',[1,1]); assignin(ws,'baseline_arch',0);
% Default arch0 deliberately fails until the runner installs real weights.
add_block('simulink/Sources/From Workspace',[modelName,'/Published record and pair'], ...
 'VariableName','baseline_inputs','SampleTime','1','Interpolate','off','OutputAfterFinalValue','Holding final value','Position',[30,110,190,150]);
add_block('simulink/Signal Attributes/Signal Specification',[modelName,'/Four numeric channels'], ...
 'Dimensions','4','OutDataTypeStr','double','Position',[220,110,310,150]);
values={'baseline_p','baseline_mu','baseline_sd','baseline_arch'};
names={'Frozen weights','Frozen training mean','Frozen training scale','Architecture selector'};
for k=1:4
 add_block('simulink/Sources/Constant',[modelName,'/',names{k}], ...
  'Value',values{k},'Position',[190,190+60*k,330,220+60*k]);
end
blockPath=[modelName,'/Normalize score and decide'];
add_block('simulink/User-Defined Functions/MATLAB Function',blockPath, ...
 'Position',[410,100,630,340]);
chart=find(sfroot,'-isa','Stateflow.EMChart','Path',blockPath);
assert(numel(chart)==1,'Baseline:Chart','Exactly one inference MATLAB Function chart is required.');
core=fileread(fullfile(versionRoot,'03_MATLAB_Kodlari','baseline_inference_numeric_v1.m'));
chart.Script=strrep(core,'function y = baseline_inference_numeric_v1(u,p,mu,sd,arch)', ...
 'function y = fcn(u,p,mu,sd,arch)');
add_block('simulink/Sinks/To Workspace',[modelName,'/Numeric trace'], ...
 'VariableName','baseline_trace','SaveFormat','Array','MaxDataPoints','inf', ...
 'Decimation','1','SampleTime','1','Position',[720,160,850,200]);
add_line(modelName,'Published record and pair/1','Four numeric channels/1');
add_line(modelName,'Four numeric channels/1','Normalize score and decide/1');
for k=1:4,add_line(modelName,[names{k},'/1'],['Normalize score and decide/',num2str(k+1)]);end
add_line(modelName,'Normalize score and decide/1','Numeric trace/1');
set_param(modelName,'Description', ...
 'A published-average recognition. Ticks are record order, not physical time. Truth/IDs are outside predictor. Default arch0 fails; run baseline_run_inference_parity_v1 to install actual frozen A weights.');
save_system(modelName,modelPath);
end
