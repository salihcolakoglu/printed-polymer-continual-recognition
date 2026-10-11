function redraw_A_v22_figures(output_folder)
% SPDX-License-Identifier: MIT
% Copyright (c) 2026 Salih Colakoglu and Mustafa Onat
% Actual execution status is recorded in figure_production_receipt_v1.json.
% Recreates the four figures from included plotting CSV files.
% No training, MAT-file loading, Simulink run, or new physical measurement.
% Original plotted data are read only. Existing image outputs are not replaced.
if nargin < 1, error('Supply an output folder.'); end
input_folder = fileparts(mfilename('fullpath'));
if ~exist(output_folder,'dir'), mkdir(output_folder); end
stems={'Figure_1_experiment_workflow','Figure_2_mechanical_records', ...
       'Figure_3_final_reporting_decisions','Figure_4_paired_old_task_gain'};
extensions={'.png','.pdf','.svg'};
for i=1:numel(stems)
    for j=1:numel(extensions)
        if exist(fullfile(output_folder,[stems{i} extensions{j}]),'file')
            error('Existing image output is preserved.');
        end
    end
end
seeds = [101 211 307 401 503];
seed_colors = [0 .447 .698; .835 .369 0; 0 .620 .451; .800 .475 .655; .349 .349 .349];
set(groot,'defaultAxesFontName','Arial','defaultTextFontName','Arial', ...
    'defaultAxesFontSize',9,'defaultFigureColor','w','defaultTextColor','k', ...
    'defaultAxesColor','w','defaultAxesXColor','k','defaultAxesYColor','k', ...
    'defaultLegendTextColor','k','defaultLegendColor','w');

% Figure 1: diagram of the completed source-record computation.
f = paper_figure(7.2,5.4);
box(f,[.22 .86 .56 .10],{'Published mechanical measurements', ...
    '104 averages in 26 processing-condition groups'},[.91 .94 .96],9);
xx=[.035 .355 .675];
texts={{'Training','72 averages / 18 groups'}, ...
       {'First development','16 averages / 4 groups'}, ...
       {'Reporting development','16 averages / 4 groups'}};
for j=1:3
    box(f,[xx(j) .68 .29 .105],texts{j},[.96 .97 .98],8.7);
    arrow(f,.50,.855,xx(j)+.145,.79);
end
box(f,[.10 .49 .80 .11],{'Two observed inputs only', ...
    'Tensile modulus (MPa) and indentation hardness (Shore D)'},[.91 .94 .96],9);
for x=[.18 .50 .82], arrow(f,x,.67,x,.605); end
box(f,[.10 .315 .80 .105],{'Known candidate pair supplied at every stage', ...
    'PLA+ vs ABS  ->  PLA+ vs PETG  ->  PLA+ vs PLA-CF'},[.96 .97 .98],9);
arrow(f,.50,.485,.50,.425);
box(f,[.035 .135 .445 .105],{'Sequential classifiers','Without replay / limited replay'},[.98 .95 .91],8.8);
box(f,[.52 .135 .445 .105],{'Reference comparisons', ...
    'Joint access to seen tasks / nearest seen response'},[.98 .95 .91],8.5);
arrow(f,.30,.31,.255,.245); arrow(f,.70,.31,.745,.245);
annotation(f,'textbox',[.02 .045 .96 .045],'String', ...
    'Every decision counted: correct / wrong / nonselection / failure', ...
    'EdgeColor','none','HorizontalAlignment','center','FontSize',8.7,'Color','k');
annotation(f,'textbox',[.02 .005 .96 .035],'String', ...
    'Completed source-record experiment; no new physical simulation', ...
    'EdgeColor','none','HorizontalAlignment','center','FontSize',8,'Color','k');
save_figure(f,output_folder,'Figure_1_experiment_workflow');

% Figure 2: all 104 average records; positions are not jittered.
S = readtable(fullfile(input_folder,'Figure_2_plot_data.csv'),'TextType','string');
f = paper_figure(6.6,4.5);
ax=axes(f,'Position',[.11 .14 .59 .82]); hold(ax,'on');
parts=["TRAIN" "FIT_DEVELOPMENT" "REPORTING_DEVELOPMENT"];
markers={'o','s','^'}; names={'PLA+','ABS','PETG','PLA-CF'};
for label=1:4
    for p=1:3
        use=S.label==label & S.partition==parts(p);
        scatter(ax,S.E_MPa(use),S.Hardness_ShoreD(use),36,seed_colors(label,:), ...
            markers{p},'filled','MarkerEdgeColor','w','LineWidth',.5);
    end
end
xlabel(ax,'Tensile modulus (MPa)'); ylabel(ax,'Indentation hardness (Shore D)');
grid(ax,'on'); ax.GridAlpha=.2;
h=gobjects(7,1);
for j=1:4, h(j)=plot(ax,nan,nan,'o','MarkerFaceColor',seed_colors(j,:), ...
        'MarkerEdgeColor','none','LineStyle','none'); end
for j=1:3, h(j+4)=plot(ax,nan,nan,markers{j},'MarkerFaceColor',[.45 .45 .45], ...
        'MarkerEdgeColor','none','LineStyle','none'); end
legend(ax,h,[names {'Training','First development','Reporting development'}], ...
    'Location','eastoutside','Box','off');
save_figure(f,output_folder,'Figure_2_mechanical_records');

% Figure 3: every fixed optimization seed, no optimal-seed selection.
T=readtable(fullfile(input_folder,'Figure_3_plot_data.csv'),'TextType','string');
methods=unique(T.method,'stable');
fields={'selected_correct','forced_argmax_correct','nonselection'};
titles={'(a) Correct selected decisions','(b) Correct forced decisions','(c) Nonselections'};
f=paper_figure(8.3,4.5);
offsets=[-.16 -.08 0 .08 .16];
for p=1:3
    positions=[.22 .28 .23 .58;.49 .28 .23 .58;.76 .28 .23 .58];
    ax=axes(f,'Position',positions(p,:));hold(ax,'on');
    for j=1:numel(methods)
        use=find(T.method==methods(j));
        for k=use'
            if T.seed(k)==0
                scatter(ax,T.(fields{p})(k),j,37,'k','d','filled');
            else
                q=find(seeds==T.seed(k));
                scatter(ax,T.(fields{p})(k),j+offsets(q),29,seed_colors(q,:),'filled');
            end
        end
    end
    set(ax,'YDir','reverse','YTick',1:7,'YLim',[.45 7.55], ...
        'XLim',[-.7 24.7],'XTick',0:4:24,'XGrid','on','GridAlpha',.25);
    if p==1, ax.YTickLabel=methods; else, ax.YTickLabel=[]; end
    xlabel(ax,'Count out of 24','FontSize',8);title(ax,titles{p},'FontSize',8.5);
end
annotation(f,'textbox',[.22 .025 .77 .13],'String', ...
    {'Seeds: 101 blue; 211 orange; 307 green; 401 purple; 503 grey.', ...
     'Black diamond: deterministic nearest reference.'}, ...
    'EdgeColor','none','HorizontalAlignment','center','FontSize',7.5,'Color','k');
save_figure(f,output_folder,'Figure_3_final_reporting_decisions');

% Figure 4: the exact five paired gains in manuscript Table 2.
G=readtable(fullfile(input_folder,'Figure_4_plot_data.csv'),'TextType','string');
row_names=unique(G.row,'stable'); M=zeros(4,5);
for i=1:4
    for j=1:5
        use=G.row==row_names(i) & G.seed==seeds(j);
        assert(nnz(use)==1); M(i,j)=G.forced_correct_gain(use);
    end
end
assert(isequal(M,[2 2 0 1 0;1 2 2 2 2;1 0 -1 0 -1;3 4 2 4 2]));
f=paper_figure(6.5,3.15);
ax=axes(f,'Position',[.33 .20 .48 .60]);imagesc(ax,M,[-4 4]);
% Signed, centred red-to-white-to-blue scale, without statistical intervals.
cmap=[linspace(.70,1,128)' linspace(.12,1,128)' linspace(.16,1,128)'; ...
      linspace(1,.13,128)' linspace(1,.34,128)' linspace(1,.62,128)'];
colormap(ax,cmap);
set(ax,'XTick',1:5,'XTickLabel',string(seeds),'YTick',1:4,'YTickLabel',row_names);
for i=1:4
    for j=1:5
        if abs(M(i,j))>=3, c='w';else,c='k';end
        text(ax,j,i,sprintf('%+d',M(i,j)),'HorizontalAlignment','center', ...
            'Color',c,'FontSize',10);
    end
end
xlabel(ax,'Optimization seed'); title(ax,'Final old-task gain: replay minus no replay','FontSize',10);
cb=colorbar(ax);cb.Ticks=-4:2:4;cb.Color='k';cb.Label.String='Gain (out of 16 old-task views)';
cb.Label.FontSize=8;
save_figure(f,output_folder,'Figure_4_paired_old_task_gain');
end

function box(f,position,texts,face,size)
annotation(f,'textbox',position,'String',texts,'EdgeColor',[.26 .31 .36], ...
    'BackgroundColor',face,'LineWidth',.9,'HorizontalAlignment','center', ...
    'VerticalAlignment','middle','FontSize',size,'FitBoxToText','off','Color','k');
end
function arrow(f,x1,y1,x2,y2)
annotation(f,'arrow',[x1 x2],[y1 y2],'Color',[.26 .31 .36],'LineWidth',1.05);
end
function save_figure(f,folder,stem)
set(f,'PaperPositionMode','auto','Color','w','InvertHardcopy','on');
print(f,fullfile(folder,[stem '.png']),'-dpng','-r600');
print(f,fullfile(folder,[stem '.pdf']),'-dpdf','-painters');
print(f,fullfile(folder,[stem '.svg']),'-dsvg','-painters');
close(f);
end
function f=paper_figure(w,h)
f=figure('Units','inches','Position',[1 1 w h],'Visible','off');
if isprop(f,'Theme'), f.Theme='light'; end
f.Color='w';
end
