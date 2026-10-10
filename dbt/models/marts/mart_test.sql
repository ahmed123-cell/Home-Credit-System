{{ config(
    materialized='incremental',
    unique_key='sk_id_curr'
) }}

select * from {{ ref('mart_application_features') }}
where dataset = 'test'