# Copyright Sierra

from .cancel_reservation import CancelReservation
from .update_reservation_flights import UpdateReservationFlights
from .update_reservation_passengers import UpdateReservationPassengers
from .update_reservation_payment import UpdateReservationPayment
from .update_reservation_baggage import UpdateReservationBaggage
from .transfer_to_human_agent import TransferToHumanAgent
from .get_user_details import GetUserDetails
from .get_reservation_details import GetReservationDetails
from .get_flight_details import GetFlightDetails
from .search_flights import SearchFlights
from .find_user_id_by_email import FindUserIdByEmail
from .find_user_id_by_name_zip import FindUserIdByNameZip
from .list_all_airports import ListAllAirports

# Optional extras for stability if needed, but the list below is the core 13.
from .book_reservation import BookReservation
from .calculate import Calculate
from .think import Think

ALL_TOOLS = [
    CancelReservation,
    UpdateReservationFlights,
    UpdateReservationPassengers,
    UpdateReservationPayment,
    UpdateReservationBaggage,
    TransferToHumanAgent,
    GetUserDetails,
    GetReservationDetails,
    GetFlightDetails,
    SearchFlights,
    FindUserIdByEmail,
    FindUserIdByNameZip,
    ListAllAirports,
    # Extras
    BookReservation,
    Calculate,
    Think,
]
