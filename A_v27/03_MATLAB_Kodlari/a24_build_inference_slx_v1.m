function modelPath = a24_build_inference_slx_v1(versionRoot)
% Creates a real SLX only on an authorized MATLAB+Simulink invocation.
% Uninitialized kind0 returns failure; runner installs actual frozen weights.
name='a24_A_inference_v1';
modelPath=fullfile(versionRoot,'04_Simulink_Modelleri',[name,'.slx']);
assert(~isfile(modelPath) && ~bdIsLoaded(name),'A24SLX:ExistingModel','Preserve existing SLX/model session.');
new_system(name);cleanup=onCleanup(@()close_system(name,0)); %#ok<NASGU>
set_param(name,'SolverType','Fixed-step','Solver','FixedStepDiscrete', ...
 'FixedStep','1','StartTime','0','StopTime','23','SaveTime','on', ...
 'TimeSaveName','tout','ReturnWorkspaceOutputs','on');
ws=get_param(name,'ModelWorkspace');
assignin(ws,'a24_inputs',timeseries(zeros(2,6),[0;1]));
assignin(ws,'a24_weights',zeros(52,1));assignin(ws,'a24_mean',[0,0]);
assignin(ws,'a24_scale',[1,1]);assignin(ws,'a24_dictionary',zeros(4,8));assignin(ws,'a24_kind',0);
add_block('simulink/Sources/From Workspace',[name,'/Published record and eligibility'], ...
 'VariableName','a24_inputs','SampleTime','1','Interpolate','off', ...
 'OutputAfterFinalValue','Holding final value','Position',[25,100,205,145]);
add_block('simulink/Signal Attributes/Signal Specification',[name,'/Six numeric channels'], ...
 'Dimensions','6','OutDataTypeStr','double','Position',[235,100,335,145]);
variables={'a24_weights','a24_mean','a24_scale','a24_dictionary','a24_kind'};
labels={'Frozen52 weight slots','Frozen TRAIN mean','Frozen TRAIN scale','Four fixed class codes','Decoder kind'};
for j=1:5
 add_block('simulink/Sources/Constant',[name,'/',labels{j}], ...
  'Value',variables{j},'Position',[190,175+60*j,350,210+60*j]);
end
block=[name,'/Normalize activate and decide'];
add_block('simulink/User-Defined Functions/MATLAB Function',block,'Position',[425,90,665,355]);
chart=find(sfroot,'-isa','Stateflow.EMChart','Path',block);
assert(numel(chart)==1,'A24SLX:Chart','Exactly one numeric inference chart is required.');
source=fileread(fullfile(versionRoot,'03_MATLAB_Kodlari','a24_inference_numeric_v1.m'));
chart.Script=strrep(source,'function y = a24_inference_numeric_v1(u,p,mu,sd,C,kind)', ...
 'function y = fcn(u,p,mu,sd,C,kind)');
add_block('simulink/Sinks/To Workspace',[name,'/Numeric trace'], ...
 'VariableName','a24_trace','SaveFormat','Array','MaxDataPoints','inf','Decimation','1', ...
 'SampleTime','1','Position',[740,160,870,205]);
add_line(name,'Published record and eligibility/1','Six numeric channels/1');
add_line(name,'Six numeric channels/1','Normalize activate and decide/1');
for j=1:5,add_line(name,[labels{j},'/1'],['Normalize activate and decide/',num2str(j+1)]);end
add_line(name,'Normalize activate and decide/1','Numeric trace/1');
set_param(name,'Description', ...
 'Frozen A24 recognition; tick is record order, not physical time. Six inputs: two measurements and four eligible-class controls. No true label or sample identity. kind1=negative code distance; kind2=seen-class softmax. No training or physical mechanics. Default kind0 fails until real weights are installed.');
save_system(name,modelPath);
end
