from collections import namedtuple
from docutils.parsers.rst import Directive, directives
from docutils import nodes


class bbp_table_value(nodes.Element):
    pass


class BBPTableValue(Directive):
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'type': directives.unchanged,
                   'value': directives.unchanged,
                   'comment': directives.unchanged
                   }

    def run(self):
        resultnode = bbp_table_value()
        self.options['field'] = self.arguments[0]
        resultnode.options = self.options
        return [resultnode]


BBPTableSectionInfo = namedtuple('BBPTableSectionInfo',
                                   'name, description, docname, targetid')


class bbp_table_section(nodes.General, nodes.Element):
    pass


class BBPTableSection(Directive):
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'description': directives.unchanged,
                   }

    def _make_table(self, rows):
        def create_table_row(row_cells):
            row = nodes.row()
            for cell in row_cells:
                row += nodes.entry('', nodes.paragraph(text=cell))
            return row

        header = ('Field', 'Value Type', 'Suggested Value', 'Comments' )
        colwidths = (1, 1, 2, 3)

        assert len(header) == len(colwidths)
        tgroup = nodes.tgroup(cols=len(header))
        for c in colwidths:
            tgroup += nodes.colspec(colwidth=c)
        tgroup += nodes.thead('', create_table_row(header))
        tbody = nodes.tbody()
        tgroup += tbody
        for row in rows:
            tbody += create_table_row((row.options['field'],
                                       row.options['type'],
                                       row.options['value'],
                                       row.options['comment']))

        table = nodes.table('', tgroup)
        return table

    def run(self):
        env = self.state.document.settings.env
        if not hasattr(env, 'all_bbp_table_sections'):
            env.all_bbp_table_sections = []

        name = self.arguments[0]
        description = self.options['description']
        targetid = "bbp_tablesection-%d" % env.new_serialno('bbp_table_section')

        node = nodes.Element()
        self.state.nested_parse(self.content, self.content_offset, node)

        section_info = BBPTableSectionInfo(name, description, env.docname, targetid)
        env.all_bbp_table_sections.append(section_info)

        children = []
        for child in node:
            if isinstance(child, bbp_table_value):
                children.append(child)

        resultnode = nodes.section(ids=[targetid])
        resultnode += [nodes.title(text=name),
                       nodes.paragraph(text=description),
                       self._make_table(children),
                       nodes.line()
                       ]

        return [resultnode]


class bbp_table_section_index(nodes.Element):
    pass


class BBPTableSectionIndex(Directive):
    '''create a place-holder for an index'''

    def run(self):
        return [bbp_table_section_index('')]


def process_bbp_table_section_index(app, doctree, fromdocname):
    env = app.builder.env

    for node in doctree.traverse(bbp_table_section_index):
        references = []
        for section in env.all_bbp_table_sections:
            ref = nodes.reference(section.name, section.name)
            ref['refdocname'] = section.docname
            ref['refuri'] = app.builder.get_relative_uri(
                fromdocname, section.docname) + '#' + section.targetid

            para = nodes.paragraph('', '', ref)
            item = nodes.list_item('', para, nodes.paragraph(text=section.description))
            references.append(item)

        content = nodes.bullet_list('', *references)
        node.replace_self(content)


def setup(app):
    app.add_node(bbp_table_section_index)
    app.add_directive('bbp_table_section_index', BBPTableSectionIndex)

    app.add_node(bbp_table_section)
    app.add_directive('bbp_table_section', BBPTableSection)
    app.add_config_value('bbp_table_section', {}, 'env')

    app.add_node(bbp_table_value)
    app.add_directive('bbp_table_value', BBPTableValue)

    app.connect('doctree-resolved', process_bbp_table_section_index)
