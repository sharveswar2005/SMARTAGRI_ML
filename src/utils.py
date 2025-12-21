class CropInputData:
    def __init__(
        self,
        Area: float,
        Item: str,
        Year: int,
        average_rain_fall_mm_per_year: float,
        pesticides_tonnes: float,
        avg_temp: float
    ):
        self.Area = Area
        self.Item = Item
        self.Year = Year
        self.average_rain_fall_mm_per_year = average_rain_fall_mm_per_year
        self.pesticides_tonnes = pesticides_tonnes
        self.avg_temp = avg_temp

    def to_dict(self):
        return {
            "Area": self.Area,
            "Item": self.Item,
            "Year": self.Year,
            "average_rain_fall_mm_per_year": self.average_rain_fall_mm_per_year,
            "pesticides_tonnes": self.pesticides_tonnes,
            "avg_temp": self.avg_temp,
        }
