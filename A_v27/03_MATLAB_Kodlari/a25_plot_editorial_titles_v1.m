function report = a25_plot_editorial_titles_v1(root)
% Plot existing A24 CSV summaries only; never train, refit, or rewrite tables.
% A25 editorial export: title-only change; accepted v5 source data remain unchanged.
if nargin<1, root=fileparts(fileparts(mfilename('fullpath'))); end
source=fullfile(root,'07_Sonuclar','learning_run_v1','summary_v1');
S=readtable(fullfile(source,'summary_by_seed.csv'),'TextType','string');
T=readtable(fullfile(source,'stage_metrics.csv'),'TextType','string');
assert(height(S)==110 && height(T)==1770,'a24:PlotPopulation','Unexpected summary population.');
seeds=[101 211 307 401 503];
assert(isequal(sort(unique(S.seed))',seeds),'a24:PlotSeeds','All five seeds must be retained.');
assert(all(ismember(S.partition,["REPORT","FIT"])) && ...
    all(ismember(T.partition,["REPORT","FIT"])), ...
    'a24:PlotPartition','REPORT and FIT must remain separate.');
out=fullfile(root,'09_Sekiller','learning_editorial_v1');
assert(~isfolder(out),'a24:NoOverwrite','Figure output exists; inspect instead of silently rerunning.');
mkdir(out);
allMethods=["onehot_no_negotiation","onehot_negotiated", ...
    "walsh_no_negotiation","walsh_negotiated", ...
    "sequential_no_replay","sequential_replay","joint_seen_reference"];
methodLabels={'One-hot / fixed','One-hot / negotiated', ...
    'Walsh / fixed','Walsh / negotiated','CE / no replay', ...
    'CE / replay','CE / joint seen'};
methodLegend={'One-hot: no negotiation','One-hot: negotiated', ...
    'Walsh: no negotiation','Walsh: negotiated', ...
    'CE: no replay','CE: replay','CE: joint-seen reference'};
colors=[0 .447 .698;.337 .706 .914;.835 .369 0;.902 .624 0; ...
    0 .620 .451;.800 .475 .655;.333 .333 .333];
markers={'o','s','^','d','v'}; offsets=[-.18 -.09 0 .09 .18];
streams=["TaskIncremental","ClassIncremental"];
streamTitles={'Known-pair task-incremental','Seen-class incremental'};
created={};
for partition=["REPORT","FIT"]
    fig=figure('Visible','off','Color','w','Units','inches','Position',[1 1 16 9.2]);
    cleanup=onCleanup(@()closeIfOpen(fig));
    heading(fig,sprintf('Final forced-decision accuracy | %s',partition));
    for ci=1:2
        count=4;if ci==2,count=7;end
        if ci==1,pos=[.075 .47 .39 .30];else,pos=[.57 .47 .40 .30];end
        ax=axes(fig,'Position',pos);hold(ax,'on');
        for mi=1:count
            rows=S(S.stream==streams(ci) & S.method==allMethods(mi) & S.partition==partition,:);
            assert(height(rows)==5 && isequal(sort(rows.seed)',seeds), ...
                'a24:PlotSeeds','A plotted method is missing a seed.');
            if ci==1,den=24;else,den=16;end
            assert(all(rows.final_all_n_views==den) && all(rows.final_unique_records==16) && ...
                all(rows.final_condition_groups==4),'a24:PlotDenominator','Wrong final denominator.');
            for si=1:5
                value=rows.final_all_accuracy_pct(rows.seed==seeds(si));
                scatter(ax,mi+offsets(si),value,120,colors(mi,:),markers{si},'filled', ...
                    'MarkerEdgeColor','w','LineWidth',.45);
            end
            average=mean(rows.final_all_accuracy_pct);
            plot(ax,mi+[-.24 .24],[average average],'-','Color','k','LineWidth',3.6);
        end
        set(ax,'XTick',1:count,'XTickLabel',methodLabels(1:count), ...
            'TickLabelInterpreter','none','FontName','Arial','FontSize',20, ...
            'YLim',[-3 105],'YTick',0:20:100,'XLim',[.5 count+.5], ...
            'Box','off','YGrid','on','GridAlpha',.20,'Layer','top');
        xtickangle(ax,35);
        if ci==1
            ylabel(ax,'Forced correct / evaluated views (%)','FontSize',22);
            cohort={'24 pair-conditioned views';'16 unique averages; 4 groups'};
        else
            cohort={'16 unique averages';'4 source-condition groups'};
        end
        title(ax,[{streamTitles{ci}};cohort], ...
            'FontName','Arial','FontSize',22,'FontWeight','normal','Interpreter','none');
    end
    footer(fig,[ ...
        'Points: seeds 101 (circle), 211 (square), 307 (triangle), 401 (diamond), 503 (down triangle).' newline ...
        'Black bars: means. Streams differ in information and cohorts; code/CE heads differ in size and loss.' newline ...
        'Previously inspected source averages; no independent specimen validation.'],[.05 .025 .92 .19]);
    saveThree(fig,sprintf('A25_final_accuracy_%s_v1',partition));
    clear cleanup

    fig=figure('Visible','off','Color','w','Units','inches','Position',[1 1 16 14.3]);
    cleanup=onCleanup(@()closeIfOpen(fig));
    heading(fig,sprintf('Acquisition and retention across learning stages | %s',partition));
    scopes=["current_unit","old_units"];
    lastLines=gobjects(7,1);
    lastAx=[];
    for ri=1:2
        for ci=1:2
            if ci==1,x0=.075;else,x0=.57;end
            if ri==1,y0=.65;plotH=.20;else,y0=.30;plotH=.22;end
            pos=[x0 y0 .40 plotH];
            ax=axes(fig,'Position',pos);hold(ax,'on');
            count=4;if ci==2,count=7;end
            for mi=1:count
                stageX=[];stageMean=[];
                for stage=1:3
                    rows=T(T.stream==streams(ci) & T.method==allMethods(mi) & ...
                        T.partition==partition & T.stage==stage & T.scope==scopes(ri) & T.n_views>0,:);
                    if isempty(rows),continue,end
                    assert(height(rows)==5 && isequal(sort(rows.seed)',seeds), ...
                        'a24:PlotSeeds','A plotted stage cohort is missing a seed.');
                    if ci==1
                        if ri==1,expectedN=8;else,expectedN=8*(stage-1);end
                    else
                        if ri==1
                            if stage==1,expectedN=8;else,expectedN=4;end
                        else,expectedN=4*stage;
                        end
                    end
                    assert(all(rows.n_views==expectedN),'a24:PlotDenominator','Wrong stage denominator.');
                    stageX(end+1)=stage;stageMean(end+1)=mean(rows.accuracy_pct); %#ok<AGROW>
                    for si=1:5
                        value=rows.accuracy_pct(rows.seed==seeds(si));
                        scatter(ax,stage+.22*offsets(si),value,65,colors(mi,:),markers{si}, ...
                            'filled','MarkerFaceAlpha',.24,'MarkerEdgeAlpha',.24);
                    end
                end
                line=plot(ax,stageX,stageMean,'o-','Color',colors(mi,:), ...
                    'LineWidth',3.4,'MarkerSize',7,'DisplayName',methodLegend{mi});
                if ci==2 && ri==2,lastLines(mi)=line;end
            end
            set(ax,'FontName','Arial','FontSize',22,'XLim',[.85 3.15], ...
                'XTick',1:3,'YLim',[-3 105],'YTick',0:20:100, ...
                'Box','off','XGrid','on','YGrid','on','GridAlpha',.18);
            if ci==1,ylabel(ax,'Forced accuracy (%)','FontSize',22);end
            if ri==2,xlabel(ax,'Learning stage (not physical time)','FontSize',22);end
            if ri==1,scopeTitle='Current acquisition unit';else,scopeTitle='Previously acquired units';end
            if ci==1
                if ri==1,denominator='Views: 8 / 8 / 8';else,denominator='Views: 0 / 8 / 16';end
            else
                if ri==1,denominator='Unique averages: 8 / 4 / 4';else,denominator='Unique averages: 0 / 8 / 12';end
            end
            title(ax,{streamTitles{ci},scopeTitle,denominator}, ...
                'FontName','Arial','FontSize',22,'FontWeight','normal','Interpreter','none');
            if ci==2 && ri==2,lastAx=ax;end
        end
    end
    lgd=legend(lastAx,lastLines,methodLegend,'Interpreter','none','NumColumns',3, ...
        'FontName','Arial','FontSize',20,'Box','off');
    set(lgd,'Units','normalized','Position',[.065 .125 .90 .10]);
    set(lastAx,'Position',[.57 .30 .40 .22]);
    footer(fig,[ ...
        'Lines: means; faint points: all five seeds. Future-task diagnostics excluded.' newline ...
        'CIL unit 1: classes 1 and 2. Task-conditioned views can reuse source averages.' newline ...
        'Final cohorts: TIL 24 views / 16 averages; CIL 16 averages. Four groups per partition.'], ...
        [.05 .012 .92 .08]);
    saveThree(fig,sprintf('A25_acquisition_retention_%s_v1',partition));
    clear cleanup
end
report=struct('status','COMPLETE','creator','MATLAB a25_plot_editorial_titles_v1', ...
    'source_summary',fullfile(source,'summary_by_seed.csv'), ...
    'source_stage_metrics',fullfile(source,'stage_metrics.csv'), ...
    'tables_modified',false,'new_training_runs',0,'figure_count',4, ...
    'file_count',numel(created),'files',{created},'MATLAB_version',version, ...
    'print_width_inches',6.7,'source_width_inches',16, ...
    'minimum_source_font_points',20,'minimum_print_font_points',20*6.7/16);

    function saveThree(figHandle,stem)
        inches=get(figHandle,'Position');inches=inches(3:4);
        set(figHandle,'PaperUnits','inches','PaperSize',inches, ...
            'PaperPosition',[0 0 inches],'InvertHardcopy','off');
        forceLightTheme(figHandle);
        drawnow;
        extensions={'.png','.pdf','.svg'};
        for fi=1:3
            target=fullfile(out,[stem extensions{fi}]);
            assert(~isfile(target),'a24:NoOverwrite','Figure file exists.');
            if fi==1
                print(figHandle,target,'-dpng','-r300');
            elseif fi==2
                print(figHandle,target,'-dpdf','-painters');
            else
                print(figHandle,target,'-dsvg','-painters');
            end
            created{end+1}=target; %#ok<AGROW>
        end
    end
end

function heading(fig,text)
annotation(fig,'textbox',[.04 .925 .92 .06],'String',text, ...
    'HorizontalAlignment','center','VerticalAlignment','middle','EdgeColor','none', ...
    'FontName','Arial','FontSize',28,'FontWeight','bold','Interpreter','none','Color','k');
end

function footer(fig,text,position)
annotation(fig,'textbox',position,'String',text,'EdgeColor','none', ...
    'FontName','Arial','FontSize',20,'Interpreter','none','Color','k','FitBoxToText','off', ...
    'HorizontalAlignment','left','VerticalAlignment','middle');
end

function closeIfOpen(fig)
if isgraphics(fig),close(fig);end
end

function forceLightTheme(fig)
% Fix theme inheritance explicitly; graphics remain readable on white exports.
set(fig,'Color','w');
axs=findall(fig,'Type','axes');
for k=1:numel(axs)
    set(axs(k),'Color','w','XColor','k','YColor','k','ZColor','k');
    set(get(axs(k),'Title'),'Color','k');
    set(get(axs(k),'XLabel'),'Color','k');
    set(get(axs(k),'YLabel'),'Color','k');
    set(get(axs(k),'ZLabel'),'Color','k');
end
texts=findall(fig,'Type','text');
if ~isempty(texts),set(texts,'Color','k');end
legends=findall(fig,'Type','legend');
if ~isempty(legends),set(legends,'TextColor','k','Color','w');end
end
