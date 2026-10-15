{% test no_infinite_or_nan(model, column_name) %}

    select {{ column_name }}
    from {{ model }}
    where {{ column_name }} is not null
      and (
          {{ column_name }} = 'Infinity'::double precision
          or {{ column_name }} = '-Infinity'::double precision
          or {{ column_name }} = 'NaN'::double precision
      )

{% endtest %}
