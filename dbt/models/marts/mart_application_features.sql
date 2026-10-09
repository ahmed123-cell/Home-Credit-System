{%- set app_cols = [] -%}
{%- if execute -%}
    {%- set app_cols = adapter.get_columns_in_relation(ref('stg_application_test')) | map(attribute='name') | list -%}
{%- endif -%}

{%- set sources = [
    {'model': 'int_bureau_agg', 'alias': 'b', 'flag': 'bureau',
     'counts': ['bureau_cnt', 'bureau_active_cnt', 'bureau_closed_cnt', 'bureau_bad_debt_cnt',
                'bureau_credit_type_nunique', 'bureau_cnt_last_2y', 'bureau_overdue_cnt']},
    {'model': 'int_previous_application_agg', 'alias': 'p', 'flag': 'prev',
     'counts': ['prev_cnt', 'prev_approved_cnt', 'prev_refused_cnt', 'prev_canceled_cnt', 'prev_cnt_last_1y']},
    {'model': 'int_installments_agg', 'alias': 'i', 'flag': 'inst',
     'counts': ['inst_cnt', 'inst_prev_nunique']},
    {'model': 'int_pos_cash_agg', 'alias': 'pc', 'flag': 'pos',
     'counts': ['pos_rows', 'pos_prev_nunique', 'pos_active_rows', 'pos_completed_rows']},
    {'model': 'int_credit_card_agg', 'alias': 'cc', 'flag': 'cc',
     'counts': ['cc_rows', 'cc_prev_nunique']},
] -%}

with applications as (
    select {{ app_cols | join(', ') }}, target, 'train' as dataset
    from {{ ref('stg_application_train') }}

    union all

    select {{ app_cols | join(', ') }}, null::bigint as target, 'test' as dataset
    from {{ ref('stg_application_test') }}
),

joined as (
    select
        a.*
        {%- for s in sources %},
        {%- for c in s.counts %}
        coalesce({{ s.alias }}.{{ c }}, 0) as {{ c }},
        {%- endfor %}
        ({{ s.alias }}.sk_id_curr is not null) as has_{{ s.flag }},
        {{ select_columns(ref(s.model), s.alias, s.counts) }}
        {%- endfor %}
    from applications a
    {%- for s in sources %}
    left join {{ ref(s.model) }} {{ s.alias }} on {{ s.alias }}.sk_id_curr = a.sk_id_curr
    {%- endfor %}
)

select
    j.*,

    -- age and employment
    (-days_birth)::double precision / 365.25                            as age_years,
    (-days_employed)::double precision / 365.25                         as employed_years,
    days_employed::double precision / nullif(days_birth, 0)             as employed_birth_ratio,

    -- loan affordability
    amt_credit::double precision / nullif(amt_income_total, 0)          as credit_income_ratio,
    amt_annuity::double precision / nullif(amt_income_total, 0)         as annuity_income_ratio,
    amt_credit::double precision / nullif(amt_annuity, 0)               as credit_term,
    amt_credit::double precision / nullif(amt_goods_price, 0)           as credit_goods_ratio,
    amt_income_total::double precision / nullif(cnt_fam_members, 0)     as income_per_person,

    -- external scores (least/greatest ignore NULLs)
    (coalesce(ext_source_1, 0) + coalesce(ext_source_2, 0) + coalesce(ext_source_3, 0))
        / nullif((ext_source_1 is not null)::int
               + (ext_source_2 is not null)::int
               + (ext_source_3 is not null)::int, 0)                    as ext_source_mean,
    least(ext_source_1, ext_source_2, ext_source_3)                     as ext_source_min,
    greatest(ext_source_1, ext_source_2, ext_source_3)                  as ext_source_max,
    ext_source_1 * ext_source_2 * ext_source_3                          as ext_source_prod,
    (ext_source_1 is not null)::int
        + (ext_source_2 is not null)::int
        + (ext_source_3 is not null)::int                               as ext_source_nonnull_cnt,

    -- documents provided
    {% for n in range(2, 22) -%}
    coalesce(flag_document_{{ n }}, 0){{ ' + ' if not loop.last }}
    {%- endfor %}                                                       as document_cnt,

    -- cross-source features
    bureau_debt_sum::double precision / nullif(amt_income_total, 0)     as bureau_debt_income_ratio,
    amt_credit::double precision / nullif(prev_amt_credit_mean, 0)      as credit_prev_credit_ratio
from joined j