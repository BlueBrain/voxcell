Those circuit files were copied from https://github.com/BlueBrain/MVDTool/tree/master/tests,
commit 098c197e3782b52b243f1247217c5f330e260221.
3 columns: region, inh_mini_frequency, exc_mini_frequency - were deleted from circuit.mvd3 because
they were not presented in circuit.mvd2.
3 columns: hypercolumn, layer, minicolumn - changed their dtype from int32 to int64.