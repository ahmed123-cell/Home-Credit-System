-- Validates target contract:
-- 1. All records in 'train' partition must have a non-null target (0 or 1)
-- 2. All records in 'test' partition must have a NULL target
select
    sk_id_curr,
    dataset,
    target
from {{ ref('mart_application_features') }}
where (dataset = 'train' and target is null)
   or (dataset = 'test' and target is not null)
