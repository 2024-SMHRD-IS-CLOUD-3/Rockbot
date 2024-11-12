import ResponseDto from "../response.dto";

export default interface SignInResponseDto extends ResponseDto {
    token: string;
    expirationTime: number;
    userId: string;
    name: string;
    departmentId: number;
}