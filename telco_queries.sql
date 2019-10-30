use telco_churn;

SELECT * 
FROM customers
JOIN contract_types USING (contract_type_id)
JOIN internet_service_types USING (internet_service_type_id)
JOIN payment_types using (payment_type_id);
