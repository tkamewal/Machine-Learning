import phonenumbers
from phonenumbers import geocoder
from phonenumbers import carrier
from phonenumbers import timezone

number = input("Enter the number with country code: ")
phone = phonenumbers.parse(number,"CH")
phone1 = phonenumbers.parse(number,"RO")
print(geocoder.description_for_number(phone, "en"))
print(carrier.name_for_number(phone1, "en"))
# print(timezone.time_zones_for_number(phone))
# print(pn.is_valid_number(phone))
# print(pn.is_possible_number(phone))
# print(pn.is_valid_number_for_region(phone, "IN"))
# print(pn.format_number(phone, pn.PhoneNumberFormat.INTERNATIONAL))
# print(pn.format_number(phone, pn.PhoneNumberFormat.NATIONAL))
# print(pn.format_number(phone, pn.PhoneNumberFormat.E164))
# print(pn.format_number(phone, pn.PhoneNumberFormat.RFC3966))


