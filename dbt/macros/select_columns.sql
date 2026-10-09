{% macro select_columns(relation, alias, exclude=[]) -%}
    {%- if execute -%}
        {%- set skip = exclude + ['sk_id_curr'] -%}
        {%- set cols = adapter.get_columns_in_relation(relation) | map(attribute='name') | reject('in', skip) | list -%}
        {%- for c in cols -%}
            {{ alias }}.{{ adapter.quote(c) }}{{ ", " if not loop.last }}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro %}