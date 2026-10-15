{% test in_range(model, column_name, min_value=none, max_value=none, inclusive=true) %}

    select {{ column_name }}
    from {{ model }}
    where {{ column_name }} is not null
      {% if min_value is not none and max_value is not none %}
          {% if inclusive %}
              and ({{ column_name }} < {{ min_value }} or {{ column_name }} > {{ max_value }})
          {% else %}
              and ({{ column_name }} <= {{ min_value }} or {{ column_name }} >= {{ max_value }})
          {% endif %}
      {% elif min_value is not none %}
          {% if inclusive %}
              and {{ column_name }} < {{ min_value }}
          {% else %}
              and {{ column_name }} <= {{ min_value }}
          {% endif %}
      {% elif max_value is not none %}
          {% if inclusive %}
              and {{ column_name }} > {{ max_value }}
          {% else %}
              and {{ column_name }} >= {{ max_value }}
          {% endif %}
      {% endif %}

{% endtest %}
