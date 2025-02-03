import numpy as np

from stable_gnn.visualization.equations.c_log_function import c_log_function
from stable_gnn.visualization.equations.calc_arrow_head_width import calc_arrow_head_width
from stable_gnn.visualization.equations.calc_common_tangent_radian import common_tangent_radian
from stable_gnn.visualization.equations.calc_direction import calc_direction
from stable_gnn.visualization.equations.calc_edge_center import calc_edge_center
from stable_gnn.visualization.equations.calc_edge_line_width import calculate_edge_line_width
from stable_gnn.visualization.equations.calc_font_size import calculate_font_size
from stable_gnn.visualization.equations.calc_init_position import init_position
from stable_gnn.visualization.equations.calc_polar_position import polar_position
from stable_gnn.visualization.equations.calc_rad_to_deg import rad_to_deg
from stable_gnn.visualization.equations.calc_safe_div import safe_div
from stable_gnn.visualization.equations.calc_vector_length import vector_length
from stable_gnn.visualization.equations.calc_vertex_line_width import calculate_vertex_line_width
from stable_gnn.visualization.equations.calc_vertex_size import calculate_vertex_size
from stable_gnn.visualization.equations.edge_list_to_incidence_matrix import edge_list_to_incidence_matrix
from stable_gnn.visualization.equations.radian_from_atan import radian_from_atan

edge_line_width = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
edge_list = ([(0, 7), (2, 7), (4, 9), (3, 7), (1, 8), (5, 7), (2, 3), (4, 5), (5, 6), (4, 8), (6, 9), (4, 7)],)
vertex_num = 10


def test__c_log_function() -> None:
    n = 10
    m = 10
    result = c_log_function(n, m)
    assert result == 1


def test__calc_arrow_head_width__show_arrow__true() -> None:
    result = calc_arrow_head_width(edge_line_width, True, edge_list[0])  # noqa
    expected_result = [0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.015]
    assert result == expected_result


def test__calc_arrow_head_width__show_arrow__false() -> None:
    result = calc_arrow_head_width(edge_line_width, False, edge_list[0])  # noqa
    expected_result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert result == expected_result


def test__calc_direction():
    x = 20
    y = 30
    direction = y - x
    result = calc_direction(direction)
    expected_result = 1.0
    assert result == expected_result


def test__calc_edge_center():
    h = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    position = np.array(
        [
            [-2.440960154591867, 0.3565427854258263],
            [0.19846435825980668, -2.189166650889786],
            [-1.3754432148586027, -4.816980082001959],
            [4.342400251353562, -4.7905135331849324],
            [1.7070082968601852, 1.832801833206762],
            [-2.3334066391035857, 3.673849985180323],
            [-2.780837316668171, 4.624152373243518],
            [-0.4839034236662598, 4.155618008010396],
            [-0.14250686323284745, -4.631455380423355],
            [2.787323470033595, -3.620255068345841],
        ]
    )
    result = calc_edge_center(h, position)
    expected_result = np.array(
        [
            [-1.4624317891290635, 2.2560803967181116],
            [-0.9296733192624312, -0.33068103699578133],
            [2.24716588344689, -0.8937266175695396],
            [1.9292484138436508, -0.31744776258726803],
        ]
    )
    assert np.array_equal(result, expected_result)


def test__calculate_edge_line_width():
    edge_list_length = 10
    result = calculate_edge_line_width(edge_list_length)
    expected_result = 0.9200444146293233
    assert result == expected_result


def test__calculate_font_size():
    result = calculate_font_size(vertex_num)
    expected_result = 18.09674836071919
    assert result == expected_result


def test__calculate_vertex_line_width():
    result = calculate_vertex_line_width(vertex_num)
    expected_result = 0.8187307530779818
    assert result == expected_result


def test__calculate_vertex_size():
    result = calculate_vertex_size(vertex_num)
    expected_result = 0.022360679774997897
    assert result == expected_result


def test__common_tangent_radian():
    r1 = 10
    r2 = 10
    d = 10
    result = common_tangent_radian(r1, r2, d)
    expected_result = 1.5707963267948966
    assert result == expected_result


def test__init_position():
    center = (10, 10)
    scale = 1.0
    result = init_position(vertex_num, center, scale)
    expected_result = np.array(
        [
            [10.317033666788971, 10.473209403635629],
            [9.686637253274482, 9.187637967109056],
            [9.20160766848048, 10.172898030345795],
            [9.314468793432976, 9.872094588366704],
            [10.619801415116468, 9.90088361575777],
            [10.373635877142448, 10.641411992995426],
            [10.942384227765514, 9.215646453541321],
            [10.937035157662027, 10.426337282147351],
            [9.907775324554745, 9.098855580876068],
            [10.442415791731094, 10.117668705203773],
        ]
    )
    assert np.array_equal(result, expected_result)


def test__polar_position():
    r = 10
    theta = 10
    start_point = np.array([0.617829638655483, 0.42093185430294533])
    result = polar_position(r, theta, start_point)
    expected_result = np.array([-7.772885652109041, -5.019279254590752])
    assert np.array_equal(result, expected_result)


def test__rad_to_deg():
    rad = 10
    result = rad_to_deg(rad)
    expected_result = 572.9577951308232
    assert result == expected_result


def test__safe_div():
    a = np.array(
        [
            [-0.42540539435398594, 0.5246694575654137],
            [-0.5393985785623281, -0.1685706355141836],
            [3.858277938978836, -0.7766769242403948],
            [-1.0708168814690318, -1.0508096035312997],
            [2.324913835790332, -0.7244725115897381],
            [0.11766458306295968, 2.387022690401171],
            [-0.759404671323388, -2.2687191541763108],
            [1.8449201088133338, 1.2727301210919089],
            [3.6383294738624508, 0.5290251557752954],
            [0.6135844935206698, -2.394919333929465],
            [5.651687304027954, -1.5203818895570083],
            [0.8764447370424022, -0.06495365417843457],
        ]
    )
    b = np.array(
        [
            [0.6754611678308796],
            [0.5651255486285466],
            [3.935674745079717],
            [1.5002831787741506],
            [2.435176536495549],
            [2.389920977479679],
            [2.392442696357482],
            [2.2413326770112527],
            [3.676589312912076],
            [2.4722711313118895],
            [5.85261740357284],
            [0.8788483113031834],
        ]
    )
    jitter = 0.000001
    result = safe_div(a, b, jitter)
    expected_result = np.array(
        [
            [-0.629799927240966, 0.7767573956180102],
            [-0.9544756556686332, -0.29828882435641574],
            [0.980334552239704, -0.1973427619270055],
            [-0.7137431763675264, -0.7004075086610607],
            [0.9547208594314499, -0.29750307656640085],
            [0.04923367097561708, 0.9987872874853106],
            [-0.31741812352688287, -0.9482856820898817],
            [0.8231353282518851, 0.5678452530255592],
            [0.9895936598316114, 0.14389019570866246],
            [0.2481865705381094, -0.9687122514991395],
            [0.9656683350901727, -0.2597781103252816],
            [0.9972650863296111, -0.07390769640567356],
        ]
    )
    assert np.array_equal(result, expected_result)


def test__vector_length():
    vector = np.array([0.05005852883380579, 0.5053637631489841])
    result = vector_length(vector)
    expected_result = 0.5078369712940438
    assert result == expected_result


def test__edge_list_to_incidence_matrix():
    result = edge_list_to_incidence_matrix(vertex_num, edge_list[0])
    expected_result = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    assert np.array_equal(result, expected_result)


def test__radian_from_atan__x_0_y_0():
    result = radian_from_atan(0, 0)
    expected_result = 4.71238898038469
    assert result == expected_result


def test__radian_from_atan__x_0_y_gt_0():
    result = radian_from_atan(0, 10)
    expected_result = 1.5707963267948966
    assert result == expected_result


def test__radian_from_atan__x_gt_0_y_0():
    result = radian_from_atan(10, 10)
    expected_result = 0.7853981633974483
    assert result == expected_result


def test__radian_from_atan__x_gt_0_y_gt_0():
    result = radian_from_atan(10, 10)
    expected_result = 0.7853981633974483
    assert result == expected_result


def test__radian_from_atan__x_gt_0_gt_y():
    result = radian_from_atan(20, 10)
    expected_result = 0.4636476090008061
    assert result == expected_result


def test__radian_from_atan__x_lt_0_lt_y():
    result = radian_from_atan(-10, 10)
    expected_result = 2.356194490192345
    assert result == expected_result
