-- Validates zero data leakage: no applicant (sk_id_curr) should exist in both train and test partitions
select
    sk_id_curr,
    count(distinct dataset) as dataset_count
from {{ ref('mart_application_features') }}
group by sk_id_curr
having count(distinct dataset) > 1
