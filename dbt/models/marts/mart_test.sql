{{ config(materialized='view') }}

select * from {{ ref('mart_application_features') }}
where dataset = 'test'