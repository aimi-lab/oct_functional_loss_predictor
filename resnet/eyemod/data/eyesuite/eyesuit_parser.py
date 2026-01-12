import datetime
import logging
import re
import math
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

class EyesuiteParser():
    SEPERATOR = ";"

    def __init__(self, parse_dates: bool = False):
        self.parse_dates = parse_dates

    def parse(self, path: str) -> pd.DataFrame:
        with open(path, "r") as file:
            lines = file.readlines()

        header = self._parse_header(lines[0])
        # lines can have different lengths, so we need to parse them individually, csv reader does not work 
        lines = [self._parse_line(line, header) for line in lines[1:]]        
        return pd.DataFrame(lines)

    def _parse_header(self, header: str) -> list[str]:
        header = header.split(self.SEPERATOR)
        header = [h.strip() for h in header]  # remove leading and trailing whitespaces
        header = [h.strip("{").strip("}") for h in header]  # remove curly brackets
        header = [h for h in header if re.match(r"\w", h)]  # remove empty headers (e.g. "" or new line)
        header = [h.lower() for h in header]  # convert to lower case
        return header

    def _parse_line(self, line: str, header: list[str]) -> dict:
        line_dict = self._line_to_dict(line, header)
        line_dict = self._parse_dict(line_dict, self.parse_dates)
        return line_dict        

    def _line_to_dict(self, line: str, header: list[str]) -> dict:
        line = line.split(";")
        line = [l.strip() for l in line]
        data = {}
        for i, h in enumerate(header):
            if i != len(header) - 1:
                data[h] = line[i]
            else:
                data[h] = line[i:]
        return data

    def _parse_dict(self, data: dict, parse_dates: bool = False) -> dict:
        """
        Parse the string values to their respective types.
        """

        string_keys = ["patient id", "eye", "program", "strategy"]

        if parse_dates == False:
            string_keys += ["examination", "date of birth"]

        for key, value in data.items():
            if key in string_keys:
                continue
            elif key == "posxyph1ph2nv":
                # this is parsed at the end
                continue

            elif key == "date of birth":
                data[key] = datetime.datetime.strptime(value, "%Y.%m.%d")
            elif key == "examination":
                data[key] = datetime.datetime.strptime(value, "%Y.%m.%d %H:%M:%S")

            else:
                try:
                    data[key] = float(value)
                except ValueError:
                    data[key] = float("nan")

        data = self._parse_measurement_points(data)

        return data

    def _parse_measurement_points(self, data: dict) ->dict:
        """
        Parse the measurement points from the POSXYPH1PH2NV column.
        5 values compose a single measurement point: x, y, ph1, ph2, nv
        """
        assert "posxyph1ph2nv" in data.keys(), "posxyph1ph2nv not found in data keys."
        assert "testlocenumber" in data.keys(), "testlocenumber not found in data keys."

        names = ["x", "y", "ph1", "ph2", "normative value"]
        points = data["posxyph1ph2nv"]
        n_points = data["testlocenumber"]

        if math.isnan(n_points):
            LOGGER.warning(
                f"testlocenumber is NaN for Patient {data['patient id']}, eye {data['eye']}, examination {data['examination']}. Required to parse measurement points. Setting point data to None."
            )
            for name in names:
                data[name] = None
            del data["posxyph1ph2nv"]
            return data

        else:
            n_points = int(n_points)
            for offset, name in enumerate(names):
                values = points[offset::5]
                values  = [float(v) for v in values[:n_points]]
                values  = [v / 10.0 for v in values] 
                data[name] = values

            del data["posxyph1ph2nv"]
            return data
        


if __name__ == "__main__":

    import visual_field as vf
    import numpy as np
    import matplotlib.pyplot as plt
    

    path = Path("/Volumes/SSD Dock/Datasets_raw/Privat/2024_OCT2VF_raw/eyesuite_export/000_eyesuite_export.csv")
    formatter = EyesuiteParser(path)
    df = formatter.read()
    df.to_csv("out/eyesuite_parsed.csv", index=False)


    df = pd.read_csv("out/eyesuite_parsed.csv")

    def str_to_list(s: str) -> list:
        if isinstance(s, str):
            return np.fromstring(s.strip("[]"), sep=",")
        else:
            return s

    for col in ["x", "y", "ph1", "ph2", "normative value"]:
        df[col] = df[col].apply(str_to_list)

    print(df.head())

    samlpe = df.iloc[3]

    visual_field = vf.VisualField(
        x_coordinate=np.asarray(samlpe["x"]),
        y_coordinate=np.asarray(samlpe["y"]),
        sensitivity_values=np.asarray(samlpe["ph1"]),
        normative_values=np.asarray(samlpe["normative value"]),
    )
    visual_field.plot_numbering()

    plt.show()
    print('done')

