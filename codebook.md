# AmazonHelp Intent Codebook

The taxonomy was created from recurring issue types observed in the
AmazonHelp customer-support subset.

## 1. delivery_issue

Problems concerning shipment or delivery.

Include:
- Package has not arrived
- Delivery is delayed
- Tracking/delivery status problems
- Delivery date problems
- Delivery person/carrier issues
- Package marked delivered but customer cannot find it

Exclude:
- Pure refund/return requests where delivery is not the main issue

---

## 2. order_issue

Problems concerning an order that are not primarily about physical delivery.

Include:
- Order details
- Order status
- Ordering problems
- Seller/order information
- Problems placing or modifying an order

Edge case:
If the main complaint is that the package is late or missing,
use `delivery_issue`.

---

## 3. payment_issue

Problems involving payment or charges.

Include:
- Payment failure
- Unexpected charge
- Payment method problems
- Card/payment information
- Billing problems

---

## 4. refund_return

Requests or problems involving refunds and returns.

Include:
- Refund not received
- Return request
- Return status
- Refund amount
- Returned item problems

Edge case:
If the customer is primarily complaining that an order was cancelled,
use `cancellation_issue`.

---

## 5. prime_issue

Problems specifically related to Amazon Prime services.

Include:
- Prime membership
- Prime benefits
- Prime delivery benefits
- Prime Video when the issue is primarily about the Prime service

---

## 6. account_issue

Problems involving the customer's Amazon account.

Include:
- Account access
- Account settings
- Account information
- Login/account-related problems

Security-related account problems may additionally trigger escalation.

---

## 7. product_device_issue

Problems with an Amazon device or product.

Include:
- Kindle
- Fire TV / Fire Stick
- Echo
- Device functionality
- Hardware/product availability
- Device-specific technical problems

---

## 8. cancellation_issue

Problems specifically involving cancellation.

Include:
- Cancel order
- Cancellation request
- Cancellation status
- Customer says an order should have been cancelled

---

## 9. other

Messages that do not fit the above categories confidently.

Include:
- Greetings
- Very short acknowledgements
- General complaints
- Ambiguous messages
- Multilingual/noisy messages where intent cannot be determined reliably
- Messages requiring conversation context that is not available

`other` is intentionally retained rather than forcing uncertain examples
into a more specific category.

---

# General annotation rules

1. Label the customer's primary issue.
2. Do not infer information that is not present in the message.
3. When multiple issues are present, choose the issue associated with the
   customer's main requested resolution.
4. Use `other` when there is insufficient information.
5. Delivery problems take precedence over generic order problems when the
   complaint is specifically about receiving the package.
6. Refund/return problems take precedence when the main requested action
   concerns getting money back or returning an item.