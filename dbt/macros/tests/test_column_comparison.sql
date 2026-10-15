{% test column_comparison(model, column_name, compare_column, operator='<=') %}

    select
        {{ column_name }},
        {{ compare_column }}
    from {{ model }}
    where {{ column_name }} is not null
      and {{ compare_column }} is not null
      and not ({{ column_name }} {{ operator }} {{ compare_column }})

{% endtest %}
