function model = r04nn_unpack(template,v)
validateattributes(v,{'double'},{'column','real','finite'});
[~,layout]=r04nn_pack(template); assert(numel(v)==layout(end).last,'r04nn:Dimensions','Parameter length mismatch.');
model=template;
for j=1:numel(layout)
    model.(layout(j).name)=reshape(v(layout(j).first:layout(j).last),layout(j).shape);
end
end
