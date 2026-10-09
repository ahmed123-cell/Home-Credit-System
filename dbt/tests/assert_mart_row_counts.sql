select dataset, count(*) as n
from {{ ref('mart_application_features') }}
group by dataset
having (dataset = 'train' and count(*) <> 307511)
    or (dataset = 'test'  and count(*) <> 48744)