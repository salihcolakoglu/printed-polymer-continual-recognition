function configPath = a30_replay_public_v1(packageRoot,outRoot,fold,prepareOnly)
% MIT. Prepare a portable, hash-checked replay of a frozen inference bank.
% No training. MATLAB R2026a + Simulink are required for actual replay.
% prepareOnly=true writes a fresh config without invoking Simulink.
if nargin<4,prepareOnly=false;end
assert(isscalar(fold) && isnumeric(fold) && isfinite(fold) && fold==floor(fold) && fold>=1 && fold<=5);
assert(islogical(prepareOnly) && isscalar(prepareOnly));
root=char(java.io.File(packageRoot).getCanonicalPath());
outRoot=char(java.io.File(outRoot).getCanonicalPath());
assert(isfolder(root));
assert(~strcmpi(outRoot,root) && ~startsWith(lower(outRoot),[lower(root),filesep]), ...
 'Write reproduction outputs outside the sealed package.');
map=jsondecode(fileread(fullfile(root,'A30_SIMULINK_PUBLIC_MAP_v1.json')));
assert(strcmp(map.schema,'A30_PORTABLE_SIMULINK_INPUT_MAP_V1'));
entry=map.folds(fold);assert(entry.fold==fold);
allpins=[map.code_pins(:);entry.bank;entry.inputs;entry.model_map;entry.reference_predictions];
for i=1:numel(allpins)
 rel=char(allpins(i).path);
 assert(~contains(rel,'..') && ~startsWith(rel,'/') && ~contains(rel,':'));
 p=fullfile(root,strrep(rel,'/',filesep));assert(isfile(p));
 assert(strcmp(file_sha256(p),allpins(i).sha256),'Public input/code hash mismatch.');
end
out=fullfile(outRoot,sprintf('fold_%d',fold));
assert(~isfolder(out) && ~isfile(out),'Preserve previous replay output; choose a fresh directory.');
mkdir(out);
cfg=struct('schema','A30_INFERENCE_RUN_CONFIG_V1','mode','public','fold',fold, ...
 'code_dir',fullfile(root,'03_MATLAB_Kodlari','simulink_parity_v1'), ...
 'bank_path',fullfile(root,strrep(entry.bank.path,'/',filesep)), ...
 'inputs_path',fullfile(root,strrep(entry.inputs.path,'/',filesep)), ...
 'out_dir',out,'expected_rows',entry.rows);
configPath=fullfile(out,'public_config.json');
f=fopen(configPath,'w','n','UTF-8');assert(f>=0);fprintf(f,'%s\n',jsonencode(cfg));fclose(f);
if ~prepareOnly
 oldPath=path;oldDir=pwd;g=Simulink.fileGenControl('getConfig');
 oldGen=struct('CacheFolder',g.CacheFolder,'CodeGenFolder',g.CodeGenFolder, ...
  'CodeGenFolderStructure',g.CodeGenFolderStructure);
 cleanup=onCleanup(@()restore_context(oldPath,oldDir,oldGen)); %#ok<NASGU>
 addpath(cfg.code_dir);a30_build_and_run_v1(configPath);
end
end
function h=file_sha256(p)
f=fopen(p,'rb');assert(f>=0);c=onCleanup(@()fclose(f)); %#ok<NASGU>
b=fread(f,Inf,'*uint8');md=java.security.MessageDigest.getInstance('SHA-256');md.update(typecast(b(:),'int8'));
h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
function restore_context(p,d,g)
try
 Simulink.fileGenControl('set','CacheFolder',g.CacheFolder, ...
  'CodeGenFolder',g.CodeGenFolder,'CodeGenFolderStructure',g.CodeGenFolderStructure);
catch err
 warning('A30:RestoreFileGen','Could not restore caller file-generation settings: %s',err.message);
end
path(p);cd(d);
end
